from findiff import FinDiff

dx = 1
dy = 1

d4_dx2dy2 = FinDiff((0, dx, 2), (1, dy, 2))
d2_dx2 = FinDiff(0, dx, 2)
d2_dy2 = FinDiff(1, dy, 2)
d2_dx2*d2_dy2

print(d4_dx2dy2.stencil((100,100)))
print(d2_dx2.stencil((100,100)))
#('C', 'C'): {(1, 1): 1.0, (1, 0): -2.0, (1, -1): 1.0, (0, 1): -2.0, (0, 0): 4.0, (0, -1): -2.0, (-1, 1): 1.0, (-1, 0): -2.0, (-1, -1): 1.0},