import struct
import numpy as np

# Standard Huffman tables (from JPEG specification Annex K)
DC_LUM_BITS = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
DC_LUM_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DC_CHR_BITS = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
DC_CHR_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

AC_LUM_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
AC_LUM_VALS = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

AC_CHR_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
AC_CHR_VALS = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

ZIGZAG = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]

def generate_huffman_codes(bits, vals):
    codes = {}
    code = 0
    val_idx = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            codes[vals[val_idx]] = (code, length)
            val_idx += 1
            code += 1
        code <<= 1
    return codes

DC_LUM_CODES = generate_huffman_codes(DC_LUM_BITS, DC_LUM_VALS)
DC_CHR_CODES = generate_huffman_codes(DC_CHR_BITS, DC_CHR_VALS)
AC_LUM_CODES = generate_huffman_codes(AC_LUM_BITS, AC_LUM_VALS)
AC_CHR_CODES = generate_huffman_codes(AC_CHR_BITS, AC_CHR_VALS)

class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.bit_buffer = 0
        self.bit_count = 0

    def write_bits(self, code, length):
        self.bit_buffer = (self.bit_buffer << length) | (code & ((1 << length) - 1))
        self.bit_count += length
        while self.bit_count >= 8:
            self.bit_count -= 8
            byte = (self.bit_buffer >> self.bit_count) & 0xFF
            self.buffer.append(byte)
            if byte == 0xFF:
                self.buffer.append(0x00)

    def flush(self):
        if self.bit_count > 0:
            self.write_bits((1 << (8 - self.bit_count)) - 1, 8 - self.bit_count)
        return self.buffer

def val2bits(value):
    if value == 0: return 0, 0
    abs_val = int(abs(value))
    category = abs_val.bit_length()
    bits = value if value > 0 else value - 1 + (1 << category)
    return category, bits

def encode_block(block, prev_dc, dc_codes, ac_codes, bw):
    block_flat = [0] * 64
    for i in range(8):
        for j in range(8):
            block_flat[ZIGZAG[i * 8 + j]] = int(block[i, j])

    dc = block_flat[0]
    diff = dc - prev_dc
    category, bits = val2bits(diff)
    code, length = dc_codes[category]
    bw.write_bits(code, length)
    if category > 0:
        bw.write_bits(bits, category)

    run = 0
    for i in range(1, 64):
        ac = block_flat[i]
        if ac == 0:
            run += 1
            if i == 63:
                code, length = ac_codes[0x00]
                bw.write_bits(code, length)
        else:
            while run > 15:
                code, length = ac_codes[0xF0]
                bw.write_bits(code, length)
                run -= 16
            
            category, bits = val2bits(ac)
            symbol = (run << 4) | category
            code, length = ac_codes[symbol]
            bw.write_bits(code, length)
            bw.write_bits(bits, category)
            run = 0
            
    return dc

def write_marker(f, marker):
    f.write(struct.pack('>H', marker))

def write_segment(f, marker, data):
    write_marker(f, marker)
    f.write(struct.pack('>H', len(data) + 2))
    f.write(data)

def build_dqt(table, id):
    data = bytearray([id])
    table_flat = [0] * 64
    for i in range(8):
        for j in range(8):
            table_flat[ZIGZAG[i * 8 + j]] = int(table[i, j])
    data.extend(table_flat)
    return data

def build_sof0(width, height):
    data = bytearray()
    data.append(8)
    data.extend(struct.pack('>H', height))
    data.extend(struct.pack('>H', width))
    data.append(3)
    data.append(1); data.append(0x11); data.append(0)
    data.append(2); data.append(0x11); data.append(1)
    data.append(3); data.append(0x11); data.append(1)
    return data

def build_dht():
    data = bytearray()
    data.append(0x00); data.extend(DC_LUM_BITS); data.extend(DC_LUM_VALS)
    data.append(0x10); data.extend(AC_LUM_BITS); data.extend(AC_LUM_VALS)
    data.append(0x01); data.extend(DC_CHR_BITS); data.extend(DC_CHR_VALS)
    data.append(0x11); data.extend(AC_CHR_BITS); data.extend(AC_CHR_VALS)
    return data

def build_sos():
    data = bytearray()
    data.append(3)
    data.append(1); data.append(0x00)
    data.append(2); data.append(0x11)
    data.append(3); data.append(0x11)
    data.append(0); data.append(63); data.append(0)
    return data

def save_jpeg_from_dct(filename, Y_q, Cb_q, Cr_q, width, height, luma_q, chroma_q):
    with open(filename, 'wb') as f:
        write_marker(f, 0xFFD8)
        app0 = bytearray(b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00')
        write_segment(f, 0xFFE0, app0)
        dqt = build_dqt(luma_q, 0) + build_dqt(chroma_q, 1)
        write_segment(f, 0xFFDB, dqt)
        sof0 = build_sof0(width, height)
        write_segment(f, 0xFFC0, sof0)
        dht = build_dht()
        write_segment(f, 0xFFC4, dht)
        sos = build_sos()
        write_segment(f, 0xFFDA, sos)
        
        bw = BitWriter()
        Hp, Wp = Y_q.shape
        dc_y = 0; dc_cb = 0; dc_cr = 0
        
        for row in range(0, Hp, 8):
            for col in range(0, Wp, 8):
                y_block = Y_q[row:row+8, col:col+8]
                cb_block = Cb_q[row:row+8, col:col+8]
                cr_block = Cr_q[row:row+8, col:col+8]
                
                dc_y = encode_block(y_block, dc_y, DC_LUM_CODES, AC_LUM_CODES, bw)
                dc_cb = encode_block(cb_block, dc_cb, DC_CHR_CODES, AC_CHR_CODES, bw)
                dc_cr = encode_block(cr_block, dc_cr, DC_CHR_CODES, AC_CHR_CODES, bw)
                
        f.write(bw.flush())
        write_marker(f, 0xFFD9)
