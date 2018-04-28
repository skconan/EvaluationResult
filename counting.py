'''
    File name: counting.py
    Author: skconan
    Date created: 2018/03/06
    Date last modified: 2018/03/08
    Python Version: 3.6.1
'''

import cv2
from lib import *

pts = []


def get_position(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append([x, y])


def draw_circle(img, pts):
    for pt in pts:
        cv2.circle(img, (pt[0], pt[1]), 5, (0, 255, 255), -1)
    return img


def warp_perspective(img, pts):
    rows, cols, ch = img.shape
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [300, 0], [0, 800], [300, 800]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res_img = cv2.warpPerspective(img, M, (300, 800))
    return res_img


def crop(img):
    global pts
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_position)
    while True:
        pts_redo = []
        res_img = draw_circle(img.copy(), pts)
        cv2.imshow('image', res_img)

        k = cv2.waitKey(1) & 0xff

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
       
        elif k == ord('b'):
            if len(pts) == 4:
                return warp_perspective(res_img, pts)
            else:
                print('Please select only 4 corners in the picture(s).')


def get_score(x):
    for i in range(1, 6):
        if x <= 60 * i:
            return 6 - i


def counting(img):
    global pts
    print('Counting')
    font = cv2.FONT_HERSHEY_SIMPLEX
    pts = []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_r, img_c = gray.shape

    cl = clahe_gray(gray)

    _, th = cv2.threshold(cl, 170, 255, cv2.THRESH_BINARY)
    mask = np.zeros((800,300),dtype=np.uint8)
    cv2.rectangle(mask,(2,2),(298,798),(255),-1)
    th = mask & th
    kernel = get_kernel('rect',(3,1))
    erode = cv2.erode(th,kernel)

    kernel = get_kernel('rect', (1, 3))
    erode = cv2.erode(erode, kernel)

    before_cnt = erode.copy()
    score_text = ''

    _, contours, _ = cv2.findContours(
        before_cnt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x += 2
        y += 2
        w -= 4
        h -= 4

        cv2.rectangle(before_cnt, (x, y), (x + w, y + h), (255), 2)
    _, contours, _ = cv2.findContours(
        before_cnt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x += 5
        y += 5
        w -= 10
        h -= 10

        area = w * h
        roi = th[y:y + h, x:x + w]

        if area >= 3000 or area <= 450:
            continue

        if h >= 70 or w >= 60 or w <= 30:
            continue

        if ct_one / (h * w * 1.0) >= 0.97:
            continue
        
        ct_one = np.count_nonzero(roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        score = str(get_score(x))
        score_text += ', ' + score
        cv2.putText(img, score, (x + int(w / 2), y + int(h / 2)),
                    font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('before', before_cnt)

    k = cv2.waitKey(-1) & 0xff

    return score_text


def main():
    f = open('counting.csv', 'wr+')
    text = '\n'
    sub_sampling = 0.3
    
    for i in range(1511, 1628):
        img_name = 'images\IMG_' + str(i) + '.JPG'
        print(img_name)
        img = cv2.imread(img_name, 1)
        img = cv2.resize(img, (0, 0), fx=sub_sampling, fy=sub_sampling)
        res_crop = crop(img)
        res_text = counting(res_crop)
        text += img_name + res_text + '\n'
        f.write(text)
    f.close()


if __name__ == '__main__':
    main()
