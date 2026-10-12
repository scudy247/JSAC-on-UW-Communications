"""
Side-information header (PLAN: Phases 4-5): the transmitter's quantised motion coefficients as bits in the frame's
header symbol, and their decoding at the receiver before the Doppler is known precisely.

Bits: for each Legendre coefficient k with b_k > 0 bits (allocation from sideinfo), its quantiser index (0 ..
2^b_k - 1) in b_k bits, then an 8-bit CRC. Both ends know the allocation and the ranges (they depend on the sea
state, not on the packet).
Modulation: differential BPSK between adjacent active subcarriers of the header symbol (no channel estimate needed:
the channel and a small timing offset change little from one subcarrier to the next); bit i is repeated on
subcarriers i, i + n, i + 2n, ... (n = number of coded bits), spread over the whole band, and soft-combined.
Decoding order at the receiver (chain.receive_with_header): own IMU, then the mean scale from preamble / postamble,
then the residual CFO of the header symbol, then the bits; if the CRC fails the receiver falls back to the
signal-only front end.
"""
import numpy as np

CRC_POLY = 0x07  # CRC-8 (x^8 + x^2 + x + 1)


def crc8(bits):
    reg = 0
    for b in bits:
        reg ^= (int(b) & 1) << 7
        reg = ((reg << 1) ^ CRC_POLY) & 0xFF if reg & 0x80 else (reg << 1) & 0xFF
    return np.array([(reg >> (7 - i)) & 1 for i in range(8)], int)


def pack(c, alloc, ranges):
    """Quantiser indices of the coefficients that get bits, as a bit array, followed by the CRC."""
    bits = []
    for k, b in enumerate(alloc):
        if b > 0:
            step = 2 * ranges[k] / 2 ** b
            idx = int(np.clip(np.floor((c[k] + ranges[k]) / step), 0, 2 ** b - 1))
            bits += [(idx >> (b - 1 - i)) & 1 for i in range(b)]
    bits = np.array(bits, int)
    return np.concatenate([bits, crc8(bits)])


def unpack(bits, alloc, ranges):
    """Coefficients from the bits (CRC removed by the caller); coefficients with no bits are 0."""
    c, pos = np.zeros(len(alloc)), 0
    for k, b in enumerate(alloc):
        if b > 0:
            idx = int("".join(str(int(x)) for x in bits[pos:pos + b]), 2)
            step = 2 * ranges[k] / 2 ** b
            c[k] = -ranges[k] + (idx + 0.5) * step
            pos += b
    return c


def modulate(bits, o, used=None):
    """One header OFDM symbol (frequency domain, (1, n_sc)): DBPSK across active subcarriers, bit i repeated every
    len(bits) subcarriers. The first active subcarrier is the phase reference."""
    act = np.where(o.active if used is None else used)[0]
    n_pos = len(act) - 1
    d = np.array([bits[i % len(bits)] for i in range(n_pos)])
    sym = np.concatenate([[1.0], np.cumprod(1 - 2 * d)])  # 0 -> same phase, 1 -> flip
    X = np.zeros((1, o.n_sc), complex)
    X[0, act] = sym
    return X


def demodulate(Y, n_bits, o, used=None):
    """Soft-combined DBPSK decisions for n_bits from the header symbol Y ((n_sc,) demodulated) -> bits, margin."""
    act = np.where(o.active if used is None else used)[0]
    z = Y[act[1:]] * np.conj(Y[act[:-1]])
    soft = np.zeros(n_bits)
    for i, v in enumerate(np.real(z)):
        soft[i % n_bits] += v
    return (soft < 0).astype(int), soft


def check(bits):
    """Split payload and CRC; True if the CRC matches."""
    payload, crc = bits[:-8], bits[-8:]
    return payload, bool(np.array_equal(crc8(payload), crc))
