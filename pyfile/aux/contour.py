import matplotlib.pyplot as plt
x = [1,2,3,4]
y = [1,2,3,4]
m = [[15,14,13,12],[14,12,10,8],[13,10,7,4],[12,8,4,0]]
cs = plt.contour(x,y,m, [9.5])
p1 = cs.collections[0].get_paths()[0]
plt.show()