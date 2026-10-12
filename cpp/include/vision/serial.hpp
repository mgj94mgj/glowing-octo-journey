#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vision {

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort();

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    bool open(const std::string& device, int baudrate, double timeout_seconds);
    void close();
    bool is_open() const;

    bool write(const std::vector<std::uint8_t>& bytes);
    std::vector<std::uint8_t> read_all();
    void reset_input_buffer();

private:
#ifdef _WIN32
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace vision
