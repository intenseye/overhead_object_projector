import torch
import numpy as np
from argparse import ArgumentParser
from models import RegressionModelXLarge, RegressionModelLarge, RegressionModelMedium, RegressionModelSmall, \
    RegressionModelXSmall

REP_COUNT = 100
WARM_UP_COUNT = 10


def measure_time_throughput(network_size, batch_size):
    """
    Measure the processing time elapsed in the inference operation.

    Parameters
    ----------
    network_size: str
        Path to input file
    batch_size: int
        Batch size

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Measurements
    """
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    if network_size == 'xl':
        centprojnet = RegressionModelXLarge
    elif network_size == 'l':
        centprojnet = RegressionModelLarge
    elif network_size == 'm':
        centprojnet = RegressionModelMedium
    elif network_size == 's':
        centprojnet = RegressionModelSmall
    elif network_size == 'xs':
        centprojnet = RegressionModelXSmall
    else:
        raise ValueError("Invalid network size %s" % repr(network_size))

    model = centprojnet(init_w_normal=False, projection_axis="both",
                        use_batch_norm=False, batch_momentum=0,
                        activation="relu")
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(batch_size, 4, dtype=torch.float).to(device)

    # Warm-up
    for _ in range(WARM_UP_COUNT):
        _ = model(dummy_input)

    # Actual measurement
    timings = np.zeros((REP_COUNT, 1))
    with torch.no_grad():
        for rep in range(REP_COUNT):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    avg_batch_time = np.sum(timings) / REP_COUNT
    avg_time_per_sample = avg_batch_time / batch_size
    throughput = 1.0 / avg_time_per_sample
    return avg_batch_time, avg_time_per_sample, throughput


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to measure time and throughput")

    args = parser.parse_args()
    batch_sizes = [1, 16, 256, 4096, 65536]
    network_sizes = ['xs', 's', 'm', 'l', 'xl']
    for batch_size_ in batch_sizes:
        avg_batch_times = []
        avg_time_per_samples = []
        throughputs = []
        for network_size_ in network_sizes:
            avg_batch_time, avg_time_per_sample, throughput = measure_time_throughput(network_size_, batch_size_)
            avg_batch_times.append(avg_batch_time)
            avg_time_per_samples.append(avg_time_per_sample)
            throughputs.append(throughput)

        print('Batch size-->: ', batch_size_)
        print('Network sizes-->: ', network_sizes)
        print('Average time (per batch) in millisecond -->: ', np.round(avg_batch_times, 3))
        print('Average time (per sample) in millisecond -->: ', np.round(avg_time_per_samples, 8))
        print('Average throughput in 1 millisecond -->: ', np.round(throughputs).astype(int))
        print('-------------------------------------------------')
