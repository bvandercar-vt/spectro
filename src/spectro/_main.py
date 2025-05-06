import pathlib
from typing import Optional, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy
import numpy.typing
from mutagen.mp3 import MP3
from pydub import AudioSegment
from rich.console import Console
from scipy import signal

BITRATE_TO_MAX_FREQ = {
    0: 0,
    16: 4000,
    64: 10000,
    128: 15000,
    192: 16000,
    320: 18000,
}

FilePath = Union[str, pathlib.Path]


def _get_samples(filename: FilePath):
    track = AudioSegment.from_file(filename)
    assert track.channels is not None
    out = numpy.array(track.get_array_of_samples()).reshape(-1, track.channels)
    return track, out


def get_spectrum(
    samples: numpy.typing.NDArray,
    channel: int,
    track,
    nperseg: Optional[float] = None,
    num_frequencies: Optional[int] = None,
):
    # Perhaps one can downsample this.
    # https://stackoverflow.com/q/60866162/353337
    f, t, Sxx = signal.spectrogram(
        samples[:, channel],
        fs=track.frame_rate,
        scaling="spectrum",
        mode="magnitude",
        nperseg=nperseg,
        # noverlap=noverlap
    )

    if num_frequencies is not None:
        # ditch some of the frequencies
        f_step = -(-f.shape[0] // num_frequencies)
        f = f[::f_step]
        Sxx = Sxx[::f_step]

    # Make sure all values are positive for the log scaling
    smallest_positive = numpy.min(Sxx[Sxx > 0])
    Sxx[Sxx < smallest_positive] = smallest_positive

    return f, t, Sxx


def get_max_freq(
    filename: FilePath,
    window_length_s: float = 0.05,
    # Use the first channel by default
    channel: int = 0,
) -> float:
    track, out = _get_samples(filename)

    if window_length_s is None:
        nperseg = None
    else:
        nperseg = int(round(window_length_s * track.frame_rate))

    f, t, Sxx = get_spectrum(out, channel, track, nperseg)

    # Which row surpasses the average first?
    log_Sxx = numpy.log10(Sxx)
    avg_log_Sxx = numpy.average(log_Sxx)
    count = numpy.sum(log_Sxx > avg_log_Sxx, axis=1)
    k = numpy.where(count > log_Sxx.shape[1] / 8)[0][-1]

    max_freq = f[k]
    return max_freq


def show(
    filename: str,
    min_freq: float = 1.0e-2,
    num_windows: Optional[int] = None,
    num_frequencies: Optional[int] = None,
    channel: Optional[int] = None,
    outfile: Optional[str] = None,
):
    track, out = _get_samples(filename)

    channels = range(out.shape[1]) if channel is None else [channel - 1]

    nperseg = None if num_windows is None else int(round(track.duration_seconds / num_windows * track.frame_rate))

    for i, k in enumerate(channels):
        f, t, Sxx = get_spectrum(out, k, track, nperseg, num_frequencies)

        plt.subplot(1, len(channels), i + 1)
        plt.pcolormesh(
            t,
            f,
            Sxx,
            norm=colors.LogNorm(vmin=min_freq, vmax=Sxx.max()),
            shading="auto",
        )
        plt.title(f"Channel {k + 1}")
        if k == 0:
            plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar()

    plt.gcf().suptitle(f"{filename} ({track.channels} channels, {track.frame_rate} Hz)")
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, transparent=True, bbox_inches="tight")


def check_dir(path: FilePath, **kwargs):
    path = pathlib.Path(path)
    if path.is_file():
        _check_file(path, **kwargs)
        return

    assert path.is_dir()
    for p in path.glob("**/*"):
        if p.suffix in [".mp3", ".wav", ".flac"]:
            _check_file(p, **kwargs)


def check_file(filename: FilePath, **kwargs):
    filename = pathlib.Path(filename)
    max_freq = get_max_freq(filename, **kwargs)

    console = Console() # colored text to terminal: https://stackoverflow.com/a/287944/353337
    
    good = False

    def check_and_log(freq_threshold: int, addl_log_str: str = ''):
        nonlocal good
        if max_freq > freq_threshold:
            console.print(f"[green]{filename} seems good{addl_log_str}.")
            good = True
        else:
            console.print(
                f"[red]{filename} is {filename.suffix.upper()}{addl_log_str}, but has max frequency about {max_freq:.0f} Hz. Check with spectro show."
            )
            good = False
            
    if filename.suffix in [".wav", ".flac"]:
        check_and_log(freq_threshold=19000)
    elif filename.suffix == ".mp3":
        mp3_file = MP3(filename)
        bitrate = int(mp3_file.info.bitrate / 1000)  # type: ignore
        for key, val in BITRATE_TO_MAX_FREQ.items():
            if bitrate < key:
                break
            expected_max_freq = val

        check_and_log(freq_threshold=expected_max_freq, addl_log_str=f" [{bitrate} kbps]")
    else:
        console.print(f"[italic]Don't know what to expect for {filename} extension {filename.suffix}.")

    return good, max_freq
