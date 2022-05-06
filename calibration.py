import cv2
import numpy as np
import os
import glob

class Calibration:

    def __init__(self, row, column, dir):
        """
        row: number of corners in rows
        column: number of corners in columns
        dir: image directory to calibrate
        """
        self.row=row
        self.column = column
        self.dir = dir

    def calib(self):
        global row, column, dir
        row=self.row
        column = self.column
        dir = self.dir

        # 체커보드의 차원 정의
        CHECKERBOARD = (row,column) # 체커보드 행과 열당 내부 코너 수
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
        objpoints = []

        # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
        imgpoints = [] 

        # 3D 점의 세계 좌표 정의
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
        images = glob.glob(f'./{dir}/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            img=cv2.resize(img,(int(img.shape[1]*0.3),int(img.shape[0]*0.3)),interpolation=cv2.INTER_LINEAR)

            # 그레이 스케일로 변환
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # 체커보드 코너 찾기
            # 이미지에서 원하는 개수의 코너가 발견되면 ret = true
            ret, corners = cv2.findChessboardCorners(gray,
                                                    CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # 원하는 개수의 코너가 감지되면,
            # 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
            if ret == True:
                objpoints.append(objp)
                # 주어진 2D 점에 대한 픽셀 좌표 미세조정
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # 코너 그리기 및 표시
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
                

            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            
        cv2.destroyAllWindows()

        h,w = img.shape[:2] # 480, 640

        # 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('calibresult.png',dst)

        tot_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        print("total calibration error: ", tot_error/len(objpoints))

        return mtx, dist, criteria

    def draw( img, corners, imgpts):
        corner = tuple(corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
        return img

    def draw_axis(self,mtx,dist,criteria):
        
        CHECKERBOARD = (row,column) 
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        for fname in glob.glob(f'{dir}/*.jpg'):
            img = cv2.imread(fname)
            img=cv2.resize(img,(int(img.shape[1]*0.3),int(img.shape[0]*0.3)),interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray,
                                                    CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = Calibration.draw(img,corners2,imgpts)
                cv2.imshow('img',img)
                k = cv2.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv2.imwrite(fname[:6]+'.png', img)
                if k == ord('q'):
                    break         
                
                cv2.destroyAllWindows()


sample=Calibration(8,6,'images')
mtx, dist, criteria= sample.calib()
sample.draw_axis(mtx,dist,criteria)