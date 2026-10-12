"""Build TensorRT engines from ONNX files.

TensorRT 11 uses strongly typed networks: the ONNX model determines precision.
Use an FP32 ONNX for an FP32 engine, or a mixed-precision ONNX (for example,
one produced by NVIDIA ModelOpt AutoCast) for an FP16/mixed engine.
"""

import argparse
from pathlib import Path
from sys import platform

from utils.general import LOGGER, check_requirements, check_version, colorstr


def convert_onnx_to_mixed_fp16(source_path, output_path):
    """Convert internal FP32 tensors to FP16 while retaining FP32 model I/O.

    TensorRT 11 requires types to be declared in the ONNX graph.  Retaining
    FP32 bindings avoids changes to YOLOv5Detector's existing input/output
    handling, while the inserted casts allow TensorRT to use FP16 internally.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            'FP16 ONNX conversion requires onnxconverter-common; install it with '\
            '`python -m pip install --user onnxconverter-common`.'
        ) from exc

    source_path, output_path = Path(source_path), Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f'ONNX file not found: {source_path}')
    LOGGER.info(f'Converting {source_path} to mixed FP16 ONNX: {output_path}')
    model = onnx.load(str(source_path))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(output_path))
    return output_path


def export_engine(onnx_path, engine_path=None, workspace=4, verbose=False,
                  prefix=colorstr('TensorRT:')):
    """Build one strongly typed TensorRT engine from an ONNX model."""
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    check_version(trt.__version__, '8.0.0', hard=True)
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path) if engine_path else onnx_path.with_suffix('.engine')
    if not onnx_path.is_file():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')

    LOGGER.info(f'\n{prefix} building {engine_path} with TensorRT {trt.__version__}...')
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace) << 30)

    # TensorRT 11 uses strongly typed, explicit-batch networks by default.
    explicit_batch_flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    network_flags = 0 if explicit_batch_flag is None else 1 << int(explicit_batch_flag)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'failed to parse ONNX file {onnx_path}:\n{errors}')

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        LOGGER.info(f'{prefix} input "{tensor.name}" shape={tensor.shape}, dtype={tensor.dtype}')
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        LOGGER.info(f'{prefix} output "{tensor.name}" shape={tensor.shape}, dtype={tensor.dtype}')

    LOGGER.info(f'{prefix} building with the precision declared by the ONNX model...')

    # TensorRT 10+ returns serialized engine bytes directly; older releases
    # retain build_engine().  Supporting both keeps this utility reusable.
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f'failed to build TensorRT engine from {onnx_path}')
        engine_path.write_bytes(bytes(serialized_engine))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError(f'failed to build TensorRT engine from {onnx_path}')
        with engine:
            engine_path.write_bytes(engine.serialize())

    LOGGER.info(f'{prefix} saved {engine_path}')
    return str(engine_path)


def main():
    parser = argparse.ArgumentParser(description='Build a TensorRT 11 engine from an ONNX model')
    parser.add_argument('--onnx', required=True, help='source ONNX model path')
    parser.add_argument('--engine', help='output TensorRT engine path (default: replace .onnx with .engine)')
    parser.add_argument('--fp16-onnx', help='write a mixed FP16 ONNX here before building the engine')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT workspace size in GiB')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    onnx_path = convert_onnx_to_mixed_fp16(args.onnx, args.fp16_onnx) if args.fp16_onnx else args.onnx
    export_engine(onnx_path, args.engine, workspace=args.workspace, verbose=args.verbose)


if __name__ == '__main__':
    main()
