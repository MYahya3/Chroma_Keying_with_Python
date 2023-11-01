import cv2
import numpy as np

def nothing(x):
    pass
def set_color_cast(x):
    global color_cast
    color_cast = x


cv2.namedWindow("Slider")
cv2.resizeWindow("Slider", 350, 350)
backGround = 0

cv2.createTrackbar("L - H", "Slider", 0, 179, nothing)
cv2.createTrackbar("U - H", "Slider", 179, 179, nothing)
cv2.createTrackbar("L - S", "Slider", 0, 255, nothing)
cv2.createTrackbar("U - S", "Slider", 255, 255, nothing)
cv2.createTrackbar("L - V", "Slider", 0, 255, nothing)
cv2.createTrackbar("U - V", "Slider", 255, 255, nothing)
cv2.createTrackbar("Gaussian Blur", "Slider", 0, 50, nothing)
cv2.createTrackbar("Change Background", "Slider", 0, 1, set_color_cast)


cap = cv2.VideoCapture("input/foreground.mp4")
cap2 = cv2.VideoCapture("output/background.mp4")

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter.fourcc(*'.mp4'), fps, (1920, 1080))

start = 0

while True:
    ret, image = cap.read()
    ret2 , img_bg = cap2.read()

    if ret == None:
        break

    if ret2 == None:
        break
    else:
        img_bg = cv2.resize(img_bg, (image.shape[1], image.shape[0]))

    l_h = cv2.getTrackbarPos("L - H", "Slider")
    u_h = cv2.getTrackbarPos("U - H", "Slider")
    l_s = cv2.getTrackbarPos("L - S", "Slider")
    u_s = cv2.getTrackbarPos("U - S", "Slider")
    l_v = cv2.getTrackbarPos("L - V", "Slider")
    u_v = cv2.getTrackbarPos("U - V", "Slider")

    blur_slider_value = cv2.getTrackbarPos("Gaussian Blur", "Slider")
    kernel_size = 2 * blur_slider_value + 1  # Ensure an odd-sized kernel

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    HSV = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    lw_g = np.array([l_h, l_s, l_v])
    up_g = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(HSV, lw_g, up_g)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    backGround = cv2.getTrackbarPos("Change Background", "Slider")
    if backGround > 0:
        start = 1

        res = cv2.bitwise_and(img_bg, img_bg, mask=mask)
        result = cv2.add(res, result)
        print(result.shape)
        # out.write(result)

    # result = cv2.bitwise_or(img_bg,fg)
    stack = np.hstack([cv2.resize(image,(0,0), fx=0.3, fy=0.5),cv2.resize(result,(0,0), fx=0.3, fy=0.5)])
    cv2.imshow("Chroma Keying", stack)

    key = cv2.waitKey(start) & 0xFF
    if key == 27:  # Exit when the 'Esc' key is pressed
        break

# out.release()
cv2.destroyAllWindows()
