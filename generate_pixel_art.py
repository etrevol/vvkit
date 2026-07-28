import os

COLORS = {
    '.': 'none',
    '#': '#ff1493',        # Deep Pink (Outline / Main color)
    'W': '#ffffff',        # White (Text Highlights / Eyes)
    'B': '#000000',        # Black (Fills)
    'D': '#ff69b4',        # Hot Pink (Lighter accents)
    'G': '#ff1493',        # Deep Pink
    'R': '#ff1493',        # Deep Pink (Blush)
    'Y': '#ff69b4',        # Hot Pink
    'X': '#000000',        # Pure Black (Background blocks)
    'E': '#1a0510',        # Very dark pinkish-black (Grid lines)
}

def render_svg(ascii_art, filename, pixel_size=16):
    lines = [line for line in ascii_art.strip("\n").split('\n')]
    if not lines:
        return
    height = len(lines)
    width = max(len(line) for line in lines)
    svg_width = width * pixel_size
    svg_height = height * pixel_size
    
    svg_content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">',
        '  <style>rect { shape-rendering: crispEdges; }</style>'
    ]
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in COLORS and COLORS[char] != 'none':
                svg_content.append(
                    f'  <rect x="{x * pixel_size}" y="{y * pixel_size}" width="{pixel_size}" height="{pixel_size}" fill="{COLORS[char]}" />'
                )
    
    svg_content.append('</svg>')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))

wordmark = """
............W.........W.
............W.....D...W.
............W.........W.
W...W.W...W.W...W.W..WWW
W...W.W...W.W..W..W...W.
.W.W...W.W..WWW...W...W.
.W.W...W.W..W..W..W...W.
..W.....W...W...W.W....W
"""

cute_head = """
...............
..#.........#..
..##.......##..
..#D#.....#D#..
..#DD#...#DD#..
..#BB#####BB#..
.#BBBBBBBBBBB#.
.#BBWWBBBWWBB#.
.#BBW#BDBW#BB#.
.#RRBBBBBBBRR#.
..#BBBBBBBBB#..
...#########...
...............
...............
"""

def make_banner(mascot_lines_str):
    banner = []
    for y in range(25):
        row = ""
        for x in range(75):
            if x % 5 == 0 and y % 5 == 0:
                row += "E"
            elif x % 5 == 0 or y % 5 == 0:
                row += "X"
            else:
                row += "."
        banner.append(list(row))

    wm_lines = [l for l in wordmark.strip("\n").split("\n")]
    wm_h = len(wm_lines)
    wm_w = len(wm_lines[0])
    offset_x = 25
    offset_y = 9

    for dy in range(wm_h):
        for dx in range(wm_w):
            if wm_lines[dy][dx] != '.':
                banner[offset_y + dy][offset_x + dx] = wm_lines[dy][dx]

    m_lines = [l for l in mascot_lines_str.strip("\n").split("\n")]
    m_h = len(m_lines)
    m_w = max(len(l) for l in m_lines) if m_lines else 0
    m_ox = 5
    m_oy = 4

    for dy in range(m_h):
        for dx in range(len(m_lines[dy])):
            if m_lines[dy][dx] != '.':
                banner[m_oy + dy][m_ox + dx] = m_lines[dy][dx]

    return "\n".join("".join(row) for row in banner)

if __name__ == "__main__":
    out_dir = "design/assets"
    
    render_svg(wordmark, os.path.join(out_dir, "wordmark.svg"), pixel_size=16)
    render_svg(cute_head, os.path.join(out_dir, "mascot.svg"), pixel_size=16)
    
    banner_str = make_banner(cute_head)
    render_svg(banner_str, os.path.join(out_dir, "banner.svg"), pixel_size=16)
    
    print(f"Generated SVGs in {out_dir}")
