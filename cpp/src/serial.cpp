#include "vision/serial.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace vision {
namespace {

#ifndef _WIN32

speed_t to_baud_rate(int baudrate) {
    switch (baudrate) {
        case 9600:
            return B9600;
        case 19200:
            return B19200;
        case 38400:
            return B38400;
        case 57600:
            return B57600;
        case 115200:
            return B115200;
        case 230400:
            return B230400;
        case 460800:
            return B460800;
        case 921600:
            return B921600;
        default:
            throw std::runtime_error("unsupported serial baudrate: " + std::to_string(baudrate));
    }
}

#endif

}  // namespace

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baudrate, double timeout_seconds) {
    close();
#ifdef _WIN32
    std::string path = device;
    if (path.rfind("\\\\.\\", 0) != 0 && path.rfind("COM", 0) == 0) {
        path = "\\\\.\\" + path;
    }
    HANDLE handle = CreateFileA(
        path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (handle == INVALID_HANDLE_VALUE) {
        return false;
    }
    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(handle, &dcb)) {
        CloseHandle(handle);
        return false;
    }
    dcb.BaudRate = static_cast<DWORD>(baudrate);
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    if (!SetCommState(handle, &dcb)) {
        CloseHandle(handle);
        return false;
    }
    COMMTIMEOUTS timeouts{};
    const DWORD timeout_ms = static_cast<DWORD>(std::max(1.0, timeout_seconds * 1000.0));
    timeouts.ReadIntervalTimeout = MAXDWORD;
    timeouts.ReadTotalTimeoutConstant = timeout_ms;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = timeout_ms;
    timeouts.WriteTotalTimeoutMultiplier = 0;
    if (!SetCommTimeouts(handle, &timeouts)) {
        CloseHandle(handle);
        return false;
    }
    handle_ = handle;
    return true;
#else
    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        fd_ = -1;
        return false;
    }

    termios options{};
    if (tcgetattr(fd_, &options) != 0) {
        close();
        return false;
    }
    cfmakeraw(&options);
    const speed_t speed = to_baud_rate(baudrate);
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = static_cast<cc_t>(
        std::clamp(static_cast<int>(timeout_seconds * 10.0), 1, 255)
    );
    if (tcsetattr(fd_, TCSANOW, &options) != 0) {
        close();
        return false;
    }
    tcflush(fd_, TCIOFLUSH);
    return true;
#endif
}

void SerialPort::close() {
#ifdef _WIN32
    if (handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

bool SerialPort::is_open() const {
#ifdef _WIN32
    return handle_ != nullptr;
#else
    return fd_ >= 0;
#endif
}

bool SerialPort::write(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty() || !is_open()) {
        return false;
    }
#ifdef _WIN32
    DWORD written = 0;
    return WriteFile(
               static_cast<HANDLE>(handle_),
               bytes.data(),
               static_cast<DWORD>(bytes.size()),
               &written,
               nullptr
           ) &&
           written == bytes.size();
#else
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(fd_, bytes.data() + offset, bytes.size() - offset);
        if (count > 0) {
            offset += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        return false;
    }
    return true;
#endif
}

std::vector<std::uint8_t> SerialPort::read_all() {
    std::vector<std::uint8_t> output;
    if (!is_open()) {
        return output;
    }
#ifdef _WIN32
    std::array<std::uint8_t, 1024> buffer{};
    DWORD read = 0;
    while (ReadFile(
        static_cast<HANDLE>(handle_),
        buffer.data(),
        static_cast<DWORD>(buffer.size()),
        &read,
        nullptr
    ) && read > 0) {
        output.insert(output.end(), buffer.begin(), buffer.begin() + read);
        if (read < buffer.size()) {
            break;
        }
    }
#else
    std::array<std::uint8_t, 1024> buffer{};
    while (true) {
        const ssize_t count = ::read(fd_, buffer.data(), buffer.size());
        if (count > 0) {
            output.insert(output.end(), buffer.begin(), buffer.begin() + count);
            if (count < static_cast<ssize_t>(buffer.size())) {
                break;
            }
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
#endif
    return output;
}

void SerialPort::reset_input_buffer() {
    if (!is_open()) {
        return;
    }
#ifdef _WIN32
    PurgeComm(static_cast<HANDLE>(handle_), PURGE_RXCLEAR);
#else
    tcflush(fd_, TCIFLUSH);
#endif
}

}  // namespace vision
