//����SVM���2ά�ռ�������3����������    
 #include "cv.h"    
#include "highgui.h"    	   
#include<ml.h>  
#include <time.h>
#include<ctype.h>
#include <AccCtrl.h>

#include <iostream>
using namespace std;   


int main(int argc, char **argv)
{
	int size = 400;         //ͼ��ĳ��ȺͿ��
	const int s = 1000;	    //�����������ɸ��ģ�����
	int i, j, sv_num;
	IplImage *img;
	CvSVM svm = CvSVM();    //����    
	CvSVMParams param;
	CvTermCriteria criteria;//ֹͣ�����ı�׼    
	CvRNG rng = cvRNG(time(NULL));
	CvPoint pts[s];         //����1000����    
	float data[s*2];        //�������    
	int res[s];             //���������    
	CvMat data_mat, res_mat;
	CvScalar rcolor;
	const float *support;

	// (1)ͼ�������ȷ���ͳ�ʼ��
	img= cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 3);
	cvZero(img);  //ȷ���������򣬲���0(�ú�ɫ����ʼ������)

	// (2)ѧϰ���ݵ�����
	for (i= 0; i< s; i++) {
		pts[i].x= cvRandInt(&rng) % size;   //�����������ֵ    
		pts[i].y= cvRandInt(&rng) % size;   
		if (pts[i].y> 50 * cos(pts[i].x* CV_PI/ 100) + 200) {   
			cvLine(img, cvPoint(pts[i].x- 2, pts[i].y- 2), cvPoint(pts[i].x+ 2, pts[i].y+ 2), CV_RGB(255, 0, 0));   
			cvLine(img, cvPoint(pts[i].x+ 2, pts[i].y- 2), cvPoint(pts[i].x- 2, pts[i].y+ 2), CV_RGB(255, 0, 0));   
			res[i] = 1;   
		}   
		else {
			if (pts[i].x> 200) {
				cvLine(img, cvPoint(pts[i].x- 2, pts[i].y- 2), cvPoint(pts[i].x+ 2, pts[i].y+ 2), CV_RGB(0, 255, 0));   
				cvLine(img, cvPoint(pts[i].x+ 2, pts[i].y- 2), cvPoint(pts[i].x- 2, pts[i].y+ 2), CV_RGB(0, 255, 0));   
				res[i] = 2;
			}
			else {
				cvLine(img, cvPoint(pts[i].x- 2, pts[i].y- 2), cvPoint(pts[i].x+ 2, pts[i].y+ 2), CV_RGB(0, 0, 255));   
				cvLine(img, cvPoint(pts[i].x+ 2, pts[i].y- 2), cvPoint(pts[i].x- 2, pts[i].y+ 2), CV_RGB(0, 0, 255));   
				res[i] = 3;   
			}   
		}   
	}   //����2ά���ѵ�����ݣ�������ֵ����CvPoint�������͵�����pts[ ]�С�    

	// (3)ѧϰ���ݵ���ʾ    
	cvNamedWindow("SVM", CV_WINDOW_AUTOSIZE);   
	cvShowImage("SVM", img);   
	cvWaitKey(0);   

	// (4)ѧϰ����������    
	for (i= 0; i< s; i++) {   
		data[i* 2] = float (pts[i].x) / size;   
		data[i* 2 + 1] = float (pts[i].y) / size;   
	}   
	cvInitMatHeader(&data_mat, s, 2, CV_32FC1, data);  //��ʼ������data_mat 
	cvInitMatHeader(&res_mat, s, 1, CV_32SC1, res);   
	criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);   
	param= CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);   
	/*  
	SVM���ࣺCvSVM::C_SVC  
	Kernel�����ࣺCvSVM::RBF  
	degree��10.0���˴β�ʹ�ã�  
	gamma��8.0  
	coef0��1.0���˴β�ʹ�ã�  
	C��10.0  
	nu��0.5���˴β�ʹ�ã�  
	p��0.1���˴β�ʹ�ã�  
	Ȼ���ѵ���������滯����������CvMat�͵������  
	*/   


	//����������(5)SVMѧϰ�������������    
	svm.train(&data_mat, &res_mat, NULL, NULL, param);//��    
	//�������ѵ�����ݺ�ȷ����ѧϰ����,����SVMѧϰ�����        

	// (6)ѧϰ����Ļ�ͼ    
	for (i= 0; i< size; i++) {   
		for (j= 0; j< size; j++) {   
			CvMat m;   
			float ret = 0.0;   
			float a[] = { float (j) / size, float (i) / size };   
			cvInitMatHeader(&m, 1, 2, CV_32FC1, a);   
			ret= svm.predict(&m);   
			switch ((int) ret) {   
			case 1:   
				rcolor= CV_RGB(100, 0, 0);   
				break;   
			case 2:   
				rcolor= CV_RGB(0, 100, 0);   
				break;   
			case 3:   
				rcolor= CV_RGB(0, 0, 100);   
				break;   
			}   
			cvSet2D(img, i, j, rcolor);   
		}   
	}   //Ϊ����ʾѧϰ�����ͨ������ͼ���������������(��������)�����з��ࡣȻ������������������ȼ�����ɫ��ͼ��    

	// (7)ѵ�����ݵ��ٻ���    
	for (i= 0; i< s; i++) {   
		CvScalar rcolor;   
		switch (res[i]) {   
		case 1:   
			rcolor= CV_RGB(255, 0, 0);   
			break;   
		case 2:   
			rcolor= CV_RGB(0, 255, 0);   
			break;   
		case 3:   
			rcolor= CV_RGB(0, 0, 255);   
			break;   
		}   
		cvLine(img, cvPoint(pts[i].x- 2, pts[i].y- 2), cvPoint(pts[i].x+ 2, pts[i].y+ 2), rcolor);   
		cvLine(img, cvPoint(pts[i].x+ 2, pts[i].y- 2), cvPoint(pts[i].x- 2, pts[i].y+ 2), rcolor);   
	}   //��ѵ�������ڽ��ͼ�����ظ��Ļ��Ƴ�����    

	// (8)֧�������Ļ���    
	sv_num= svm.get_support_vector_count();   
	for (i= 0; i< sv_num; i++) {   
		support = svm.get_support_vector(i);   
		cvCircle(img, cvPoint((int) (support[0] * size), (int) (support[1] * size)), 5, CV_RGB(200, 200, 200));   
	}   
	//�ð�ɫ��ԲȦ��֧����������ǡ�    

	// (9)ͼ�����ʾ     
	cvNamedWindow("SVM", CV_WINDOW_AUTOSIZE);   
	cvShowImage("SVM", img);   
	cvWaitKey(0);   
	cvDestroyWindow("SVM");   
	cvReleaseImage(&img);   
	return 0;   
	//��ʾʵ�ʴ�������ͼ��ֱ��ĳ����������Ϊֹ��    
}  
