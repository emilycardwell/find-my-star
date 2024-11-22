import math
import cv2
import numpy as np
from itertools import combinations
from scipy.spatial import KDTree


### FUNCTIONS ###


def process_image(image, cutoff=225):
    grey_img = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE)
    sq_img = grey_img[:min(grey_img.shape), :min(grey_img.shape)]
    sm_img = cv2.resize(sq_img, dsize=(1000,1000))
    cut_img = sm_img.copy()
    cut_img[cut_img<cutoff] = 0
    _, bi_img = cv2.threshold(cut_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bi_img


def get_test_centroids(contours):
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


def get_triangulators(centroids):
    star_tree = KDTree(centroids, )
    triangles = []
    central_stars = []
    for centroid in centroids:
        _, central_idxs = star_tree.query(centroid, k=50)
        center_star = centroids[central_idxs[0]]
        outer_stars = [centroids[i] for i in central_idxs[1:]]

        emcomp_tri = check_areas(center_star, outer_stars)
        if emcomp_tri:
            triangles.append(emcomp_tri)
            central_stars.append(centroid)

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

        inner = sorted([angle1, angle2, angle3])

        angle4 = math.acos((AB**2 + AP**2 - BP**2) / (2 * AB * AP)) * 180 / math.pi
        try:
            angle5 = math.acos((AB**2 + BP**2 - AP**2) / (2 * AB * BP)) * 180 / math.pi
        except ValueError:
            angle5 = 0
        angle6 = math.acos((BC**2 + BP**2 - CP**2) / (2 * BC * BP)) * 180 / math.pi
        angle7 = math.acos((BC**2 + CP**2 - BP**2) / (2 * BC * CP)) * 180 / math.pi
        angle8 = math.acos((AC**2 + AP**2 - CP**2) / (2 * AC * AP)) * 180 / math.pi
        angle9 = math.acos((AC**2 + CP**2 - AP**2) / (2 * AC * CP)) * 180 / math.pi

        outer = sorted([angle4, angle5, angle6, angle7, angle8, angle9])

        angles.append(inner + outer)

    return angles


# ### MAIN ###


def main(img):
    proc_img = process_image(img)
    contours, _ = cv2.findContours(proc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = get_test_centroids(contours)
    triangles, central_stars = get_triangulators(centroids)
    angles = get_angles(triangles, central_stars)
    return central_stars, triangles, angles


if __name__ == '__main__':
    ref_angles = main('data/reference/star_chart_ireland_2024-10-11_01-00.jpg')
    test_angles = main('data/testing/astro1.jpg')
