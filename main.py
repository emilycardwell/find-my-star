import math
import cv2
import numpy as np
from itertools import combinations
from scipy.spatial import KDTree


### FUNCTIONS ###


def process_image(image):
    grey_img = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE)
    sq_img = grey_img[:min(grey_img.shape), :min(grey_img.shape)]
    sm_img = cv2.resize(sq_img, dsize=(1000,1000))
    cut_img = sm_img.copy()
    cut_img[cut_img<200] = 0
    _, bi_img = cv2.threshold(cut_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bi_img


def get_centroids(contours):
    contours = [[[pt[0][0] for pt in cont],[pt[0][1] for pt in cont]] for cont in contours]

    centroids = []
    for cont in contours:
        xs = cont[0]
        ys = cont[1]

        xroll = np.roll(xs,1)
        yroll = np.roll(ys,1)
        poly_area = abs(np.dot(xs, yroll) - np.dot(ys, xroll))/2

        if poly_area == 0:
            cx = np.round(np.mean(xs), 2)
            cy = np.round(np.mean(ys), 2)
        else:
            cx = np.round(np.sum((xs + xroll) * (xs * yroll - xroll * ys))/(6 * poly_area), 2)
            cy = np.round(np.sum((ys + yroll) * (xs * yroll - xroll * ys))/(6 * poly_area), 2)

        centroids.append((cx, cy))

    return centroids


def area(X, Y, n):
    area = 0.0
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i

    return abs(area / 2.0)


def get_triangles(center, ABC):
    PAB = [center, ABC[0], ABC[1]]
    PBC = [center, ABC[1], ABC[2]]
    PAC = [center, ABC[0], ABC[2]]

    return ABC, [PAB, PBC, PAC]


def get_coords(triangle):
    X = [x[0] for x in triangle]
    Y = [x[1] for x in triangle]

    return X, Y


def check_areas(center_star, outer_stars):
    if len(outer_stars) > 3:
        poss_triangles = list(combinations(outer_stars, 3))
    else:
        poss_triangles = [outer_stars]

    for triangle in poss_triangles:
        outer_tri, inner_tris = get_triangles(center_star, triangle)
        outer_area = area(*get_coords(outer_tri), 3)

        inner_area = 0
        for tri in inner_tris:
            inner_area += area(*get_coords(tri), 3)

        if abs(outer_area - inner_area) < 1e-9:
            return triangle


def get_distance(p1, p2):
	return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_triangulators(stars, star_tree):
    triangles = []
    central_stars = []
    for star in stars:
        _, center_idx = star_tree.query(star, k=1)
        center_star = stars[center_idx]

        _, central_idxs = star_tree.query(star, k=70)
        outer_stars = [stars[i] for i in central_idxs[1:]]

        emcomp_tri = check_areas(center_star, outer_stars)
        if emcomp_tri:
            triangles.append(emcomp_tri)
            central_stars.append(star)

    return triangles, central_stars


def get_angles(triangles, central_stars):
    angles = []
    for i in range(len(triangles)):
        A, B, C = triangles[i]
        P = central_stars[i]

        AP = get_distance(A, P)
        BP = get_distance(B, P)
        CP = get_distance(C, P)
        AB = get_distance(A, B)
        BC = get_distance(B, C)
        AC = get_distance(A, C)

        angle1 = math.acos((AP**2 + BP**2 - AB**2) / (2 * AP * BP)) * 180 / math.pi
        angle2 = math.acos((BP**2 + CP**2 - BC**2) / (2 * BP * CP)) * 180 / math.pi
        angle3 = math.acos((CP**2 + AP**2 - AC**2) / (2 * CP * AP)) * 180 / math.pi

        angle_sum = sum([angle1, angle2, angle3])
        if 360 - angle_sum > 1e-9:
            print('error', angle_sum)

        angles.append([angle1, angle2, angle3])

    return angles


### MAIN ###


def main(img):
    proc_img = process_image(img)
    contours, _ = cv2.findContours(proc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = get_centroids(contours)
    star_tree = KDTree(centroids, )
    triangles, central_stars = get_triangulators(centroids, star_tree)
    angles = get_angles(triangles, central_stars)
    return central_stars, triangles, angles


if __name__ == '__main__':
    ref_angles = main('data/reference/star_chart_ireland_2024-10-11_01-00.jpg')
    test_angles = main('data/testing/astro1.jpg')
