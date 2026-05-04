"""Solutions for Lesson 28 exercises."""


def validate_global_batch_size(batch_size: int, num_devices: int) -> bool:
    return num_devices > 0 and batch_size >= num_devices and batch_size % num_devices == 0
