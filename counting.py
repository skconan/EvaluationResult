'''
    File name: counting.py
    Author: skconan
    Date created: 2018/03/06
    Date last modified: 2018/03/08
    Python Version: 3.6.1
'''

import cv2 as cv
import numpy as np
from operator import itemgetter

pts = []
LEVEL = 5
ROWS_IGNORE = [1, 6, 7, 11, 14, 17, 20, 21, 23, 27]
TABLE_HEIGHT, TABLE_WIDTH = 800, 300


def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    elif shape == '\\':
        kernel = np.diag([1] * ksize[0])
        return np.uint8(kernel)
    elif shape == '/':
        kernel = np.fliplr(np.diag([1] * ksize[0]))
        return np.uint8(kernel)
    else:
        return None


def clahe_gray(img_gray):
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    res_gray = clahe.apply(img_gray)
    return res_gray


def get_position(event, x, y, flags, param):
    global pts
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append([x, y])


def draw_circle(img, pts):
    for pt in pts:
        cv.circle(img, (pt[0], pt[1]), 5, (0, 255, 255), -1)
    return img


def warp_perspective(img, pts):
    rows, cols, ch = img.shape
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [300, 0], [0, 800], [300, 800]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    res_img = cv.warpPerspective(img, M, (300, 800))
    return res_img


def crop(img):
    global pts
    cv.namedWindow('image')
    cv.setMouseCallback('image', get_position)
    while True:
        pts_redo = []
        res_img = draw_circle(img.copy(), pts)
        cv.imshow('image', res_img)

        k = cv.waitKey(1) & 0xff

        if k == ord('e'):
            exit(0)

        elif k == ord('z'):
            print('undo')
            if len(pts):
                val = pts.pop()
                pts_redo.append(val)

        elif k == ord('x'):
            print('redo')
            if len(pts_redo):
                val = pts_redo.pop()
                pts.append(val)

        elif k == ord('c'):
            if len(pts) == 4:
                return warp_perspective(res_img, pts)
            else:
                print('Please select only 4 corners in the picture(s).')


def get_score(x):
    global LEVEL, TABLE_WIDTH
    for i in range(1, LEVEL + 1):
        if x <= (TABLE_WIDTH / LEVEL) * i:
            return LEVEL + 1 - i


def find_table(img_bin):
    global LEVEL, ROWS_IGNORE
    row = []
    result = []

    for i in range(4):
        img_bin[:, i * 60:i * 60 + 60] += img_bin[:, -60:]
        img_bin[:, -60:] += img_bin[:, i * 60:i * 60 + 60]

    _, before_cnt = cv.threshold(img_bin, 127, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(
        before_cnt, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w * h < 120:
            print('cont.')
            continue
        x += 2
        y += 2
        w -= 4
        h -= 4
        row.append([x, y, w, h])

    row = sorted(row, key=itemgetter(1))
    tmp = []
    for r in row:
        x, y, w, h = r
        tmp.append(r)
        if len(tmp) == LEVEL:
            tmp = sorted(tmp, key=itemgetter(0))
            # print(tmp)
            result.append(tmp)
            tmp = []

    for r in ROWS_IGNORE:
        result[r - 1] = None
    for i in range(len(ROWS_IGNORE)):
        result.remove(None)
    ct = 0
    for r in result:
        for c in r:
            x, y, w, h = c
        ct += 1
    return result, img_bin


def counting(img):
    # Variable declaration
    global TABLE_HEIGHT, TABLE_WIDTH, pts
    pts = []
    thresh = 170
    text = ''
    font = cv.FONT_HERSHEY_SIMPLEX
    result_shape = (TABLE_HEIGHT, TABLE_WIDTH)

    print('Counting')

    '''
        Convert BGR to GrayScale Image
        Equalization by Contrast-limited adaptive histogram equalization (CLAHE)
        Convert GrayScale to Binary Image by thresh
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cl = clahe_gray(gray)
    _, th_main = cv.threshold(cl, thresh, 255, cv.THRESH_BINARY)

    # Add inner border to frame
    mask = np.zeros(result_shape, dtype=np.uint8)
    cv.rectangle(mask, (2, 2), (298, 798), (255), -1)
    th = mask & th_main.copy()

    # Erode for clearly check symbol and seprerate box.
    kernel = get_kernel('rect', (3, 1))
    erode1 = cv.erode(th, kernel)
    kernel = get_kernel('rect', (1, 3))
    erode2 = cv.erode(erode1, kernel)

    dist_transform = cv.distanceTransform(erode2.copy(), cv.DIST_L2, 5)
    ret, before_cnt = cv.threshold(
        dist_transform, 0.2 * dist_transform.max(), 255, 0)
    before_cnt = np.uint8(before_cnt)
    _, contours, _ = cv.findContours(
        before_cnt, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    table, img_table = find_table(before_cnt)
    th_main = th_main & img_table
    _, th_main = cv.threshold(th_main, 127, 255, cv.THRESH_BINARY)
    for row in table:
        ratio_min = 1
        col_checked = None
        for col in row:
            x, y, w, h = col
            white_black_ratio = np.count_nonzero(
                th_main[y:y + h, x:x + w]) / (w * h)

            if white_black_ratio < ratio_min:
                ratio_min = white_black_ratio
                col_checked = col
            # cv.putText(img, ("%.2f" % (white_black_ratio)), (x + int(w / 2), y + int(h / 2)),font, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            # cv.imshow('sd',th_main[y:y + h, x:x + w])
            # cv.waitKey(-1)
        if col_checked is not None:
            x, y, w, h = col_checked
            text += str(get_score(x)) + ', '
        else:
            text += '0, '
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)

    # cv.imshow('img', img)
    # cv.imshow('before', before_cnt)
    # cv.imshow('th', th_main)
    # cv.imshow('erode1', erode1)
    # cv.imshow('erode2', erode2)

    # k = cv.waitKey(-1) & 0xff

    return text


def main():
    print('<'*10+' How to use '+10*'>')
    print("1) Put the filename for collect output (Don't specific file type because output specific csv file)")
    print("2) Put the prefix filename")
    print("3) Put the range of file")
    print("4) Choose four point for crop table and pree 'C' to continue the next images")
    print("5) If u want to undo press 'z'")
    print("5) If u want to redo press 'x'")
    print("6) If u want to exit press 'e' 2 times")

    filename = str(input('Please input file name: '))
    f = open(filename + '.csv', 'a')
    text = '\n'
    sub_sampling = 0.3

    for i in range(1511, 1628):
        img_name = 'images\IMG_' + str(i) + '.JPG'
        print(img_name)
        img = cv.imread(img_name, 1)
        img = cv.resize(img, (0, 0), fx=sub_sampling, fy=sub_sampling)
        res_crop = crop(img)
        res_text = counting(res_crop)
        text = img_name + ', ' + res_text + '\n'
        f.write(text)
    f.close()


if __name__ == '__main__':
    main()
