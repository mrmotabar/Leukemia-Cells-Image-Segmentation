import cv2
from skimage import io
import numpy as np
from copy import deepcopy
from random import random, randint
from math import floor, ceil
#from matplotlib import pyplot as plt


class particle:
    def __init__(self, n_clusters, image, colors, mode):
        if mode == 0:
            self.x = np.random.rand(n_clusters, 3)
        elif mode == 1:
            self.x = np.zeros((n_clusters, 3))
            for i in range(n_clusters):
                ind = randint(0, len(colors) - 1)
                for j in range(3):
                    self.x[i][j] = colors[ind][j]
        elif mode == 2:
            self.x = np.array(colors)
        elif mode == 3:
            self.x = ((np.random.rand(n_clusters, 3) * 2) - 1) / \
                50 + np.array(colors)
        self.n_clusters = n_clusters
        self.v = ((np.random.rand(n_clusters, 3) * 2) - 1) / 4
        self.cost = cal_cost(self.x, image)
        self.pbest = deepcopy(self.x)
        self.pbest_cost = self.cost

    def upd(self, image):
        cost = cal_cost(self.x, image)
        self.cost = cost
        if cost < self.pbest_cost:
            self.pbest_cost = cost
            self.pbest = deepcopy(self.x)

    def normalize(self):
        for i in range(self.n_clusters):
            for j in range(3):
                if (self.x[i][j] > 1):
                    self.x[i][j] = 1
                if (self.x[i][j] < 0):
                    self.x[i][j] = 0


def dist(color_1, color_2):
    s = 0
    for i in range(len(color_1)):
        s += (color_1[i] - color_2[i]) ** 2
    return np.sqrt(s)


def find_nearest_center(par, color):
    ind = np.argmin([dist(par[i], color) for i in range(len(par))])
    return (ind, dist(par[ind], color))
    ind_min = 0
    val_min = dist(par[0], color)
    for i in range(1, len(par)):
        temp = dist(par[i], color)
        if temp < val_min:
            val_min = temp
            ind_min = i
    return (ind_min, val_min)


def segment(image, centers):
    im = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            im[i][j] = centers[find_nearest_center(centers, image[i][j])[0]]
    return im


def find_imp_colors(image):
    imp_colors = []
    r = [0 for _ in range(256)]
    g = [0 for _ in range(256)]
    b = [0 for _ in range(256)]
    for i in range(len(image)):
        for j in range(len(image[i])):
            r[floor(image[i][j][0] * 255)] += 1
            g[floor(image[i][j][1] * 255)] += 1
            b[floor(image[i][j][2] * 255)] += 1
    tops_r = []
    tops_g = []
    tops_b = []
    ra = 20
    for i in range(1, 256):
        if r[i] > 20:
            if len(tops_r) == 0:
                tops_r.append(i)
            elif i - tops_r[-1] < 20 and r[i] > r[tops_r[-1]]:
                tops_r[-1] = i
            elif i - tops_r[-1] >= 20:
                tops_r.append(i)
        if g[i] > 20:
            if len(tops_g) == 0:
                tops_g.append(i)
            elif i - tops_g[-1] < 20 and g[i] > g[tops_g[-1]]:
                tops_g[-1] = i
            elif i - tops_g[-1] >= 20:
                tops_g.append(i)
        if b[i] > 20:
            if len(tops_b) == 0:
                tops_b.append(i)
            elif i - tops_b[-1] < 20 and b[i] > b[tops_b[-1]]:
                tops_b[-1] = i
            elif i - tops_b[-1] >= 20:
                tops_b.append(i)
    for i in tops_r:
        for j in tops_g:
            for k in tops_b:
                imp_colors.append([i / 255, j / 255, k / 255])
    return imp_colors


def best_colors(imp_colors, n_clusters, image):
    n_pix_clusters = [[0, imp_colors[i]] for i in range(len(imp_colors))]
    for i in image:
        for j in i:
            n_pix_clusters[find_nearest_center(imp_colors, j)[0]][0] += 1
    n_pix_clusters.sort()
    ret = []
    for i in range(n_clusters):
        ret.append(deepcopy(n_pix_clusters[-1 - i][1]))
    return ret


def cal_cost(par, image):
    cost = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            cost += find_nearest_center(par, image[i][j])[1]
    return cost


def find_best_centers(image):
    imp_colors = find_imp_colors(image)
    n_particles = 20
    n_clusters = 5
    n_generations = 20
    w = 0.4
    c_p = 0.3
    c_g = 0.4
    particels = []
    # for i in range(n_particles // 2):
    #    particels.append(particle(n_clusters, image, imp_colors, 1))
    bc = best_colors(imp_colors, n_clusters, image)
    particels.append(particle(n_clusters, image, bc, 2))
    for i in range(0, n_particles - 1):
        particels.append(particle(n_clusters, image, bc, 3))
    g_best_ind = 0
    for i in range(1, n_particles):
        if particels[i].cost < particels[g_best_ind].cost:
            g_best_ind = i
    #print("g_best_cost = " + str(particels[g_best_ind].pbest_cost))
    gen = 0
    while gen < n_generations:
        #print("Generation : " + str(gen) + "    g_best_cost = " + str(particels[g_best_ind].pbest_cost))
        for i in range(n_particles):
            particels[i].v = w * particels[i].v + c_p * random() * (particels[i].pbest -
                                                                    particels[i].x) + c_g * random() * (particels[g_best_ind].pbest - particels[i].x)
            particels[i].x += particels[i].v
            particels[i].normalize()
            particels[i].upd(image)
            if particels[i].pbest_cost < particels[g_best_ind].pbest_cost:
                g_best_ind = i
        gen += 1
    return particels[g_best_ind].pbest


def compress(image, size):
    if size == 1:
        return image
    n, m = image.shape[: 2]
    n = ceil(n / size)
    m = ceil(m / size)
    im = np.zeros((n, m, 3))
    for i in range(len(image)):
        for j in range(len(image[i])):
            im[i // size][j // size] += image[i][j]
    for i in range(len(im)):
        for j in range(len(im[i])):
            im[i][j] /= size ** 2
    return im


def decompress(image, size):
    if size == 1:
        return image
    n, m = image.shape[: 2]
    n *= size
    m *= size
    im = np.zeros((n, m, 3))
    for i in range(len(im)):
        for j in range(len(im[i])):
            im[i][j] += image[i // size][j // size]
    return im


def improve_with_mean(par, image):
    colors = np.array([[0.0, 0.0, 0.0] for _ in par])
    colors_t = [0 for _ in par]
    for i in range(len(image)):
        for j in range(len(image[i])):
            ind = find_nearest_center(par, image[i][j])[0]
            colors[ind] += image[i][j]
            colors_t[ind] += 1
    for i in range(len(colors)):
        if colors_t[i] != 0:
            colors[i] /= colors_t[i]
    return colors


def improve_with_hill(par, image):
    cur_cost = cal_cost(par, image)
    cur = deepcopy(par)
    epsilon = 0.004
    for _ in range(10):
        for i in range((len(par) * 3)):
            r = randint(0, 2)
            nxt = deepcopy(cur)
            if r % 3 == 1:
                nxt[i // 3][i % 3] += epsilon
            elif r % 3 == 2:
                nxt[i // 3][i % 3] -= epsilon
            nxt_cost = cal_cost(nxt, image)
            if nxt_cost < cur_cost:
                cur = deepcopy(nxt)
                cur_cost = nxt_cost
    return cur


def cal_psnr(i_image, o_image):
    psnr_layer = np.array([0.0, 0.0, 0.0])
    n, m = i_image.shape[: 2]
    for i in range(n):
        for j in range(m):
            psnr_layer[0] += ((i_image[i][j][0] - o_image[i][j][0]) * 255) ** 2
            psnr_layer[1] += ((i_image[i][j][1] - o_image[i][j][1]) * 255) ** 2
            psnr_layer[2] += ((i_image[i][j][2] - o_image[i][j][2]) * 255) ** 2
    psnr_layer /= n * m
    psnr_layer = (256 ** 2) / psnr_layer
    psnr_layer = 10 * np.log10(psnr_layer)
    return psnr_layer


start = int(input())
end = int(input())
sum_cost = 0
sum_cost_255 = 0
sum_psnr = 0
for file_number in range(start, end + 1):
    image = (io.imread(".\\ALL_IDB2\\img\\Im" +
             "{:03d}".format(file_number) + "_" + str(1 - file_number // 131) + ".tif")) / 255
    im = compress(image, 4)
    best = []
    best.append(find_best_centers(im))
    best.append(improve_with_hill(best[0], im))
    best.append(improve_with_mean(best[0], image))
    best.append(improve_with_mean(best[2], image))
    cost = np.array([cal_cost(par, image) for par in best])
    best_ind = np.argmin(cost)
    psnr = cal_psnr(im, image)
    # print(cost[best_ind])
    #print(cost[best_ind] * 255)
    im = segment(image, best[best_ind])
    #print(cal_psnr(best[best_ind], im, image))
    final = cv2.hconcat((image, im))
    print(str(file_number) + " :   " + str(cost[best_ind] * 255))
    sum_cost += cost[best_ind]
    sum_cost_255 += cost[best_ind] * 255
    sum_psnr += np.sum(psnr) / 3
    f = open(".\\Results_5\\Im" + "{:03d}".format(file_number) +
             "_" + str(1 - file_number // 131) + ".txt", "w")
    f.write("Cost :" + str(cost[best_ind]) + "\n")
    f.write("Cost-255 :" + str(cost[best_ind] * 255) + "\n")
    f.write("PSNR : " + str(psnr) + "\n")
    f.write("Clusters : " + str(best[best_ind]) + "\n")
    f.close()
    io.imsave(".\\Results_5\\Im" + "{:03d}".format(file_number) +
              "_" + str(1 - file_number // 131) + ".jpg", final)
