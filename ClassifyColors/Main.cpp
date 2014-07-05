//利用SVM解决2维空间向量的3级分类问题    
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
	int size = 400;         //图像的长度和宽度
	const int s = 1000;	    //试验点个数（可更改！！）
	int i, j, sv_num;
	IplImage *img;
	CvSVM svm = CvSVM();    //★★★    
	CvSVMParams param;
	CvTermCriteria criteria;//停止迭代的标准    
	CvRNG rng = cvRNG(time(NULL));
	CvPoint pts[s];         //定义1000个点    
	float data[s*2];        //点的坐标    
	int res[s];             //点的所属类    
	CvMat data_mat, res_mat;
	CvScalar rcolor;
	const float *support;

	// (1)图像区域的确保和初始化
	img= cvCreateImage(cvSize(size, size), IPL_DEPTH_8U, 3);
	cvZero(img);  //确保画像区域，并清0(用黑色作初始化处理)

	// (2)学习数据的生成
	for (i= 0; i< s; i++) {
		pts[i].x= cvRandInt(&rng) % size;   //用随机整数赋值    
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
	}   //生成2维随机训练数据，并将其值放在CvPoint数据类型的数组pts[ ]中。    

	// (3)学习数据的显示    
	cvNamedWindow("SVM", CV_WINDOW_AUTOSIZE);   
	cvShowImage("SVM", img);   
	cvWaitKey(0);   

	// (4)学习参数的生成    
	for (i= 0; i< s; i++) {   
		data[i* 2] = float (pts[i].x) / size;   
		data[i* 2 + 1] = float (pts[i].y) / size;   
	}   
	cvInitMatHeader(&data_mat, s, 2, CV_32FC1, data);  //初始化矩阵data_mat 
	cvInitMatHeader(&res_mat, s, 1, CV_32SC1, res);   
	criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);   
	param= CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);   
	/*  
	SVM种类：CvSVM::C_SVC  
	Kernel的种类：CvSVM::RBF  
	degree：10.0（此次不使用）  
	gamma：8.0  
	coef0：1.0（此次不使用）  
	C：10.0  
	nu：0.5（此次不使用）  
	p：0.1（此次不使用）  
	然后对训练数据正规化处理，并放在CvMat型的数组里。  
	*/   


	//☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆    
	svm.train(&data_mat, &res_mat, NULL, NULL, param);//☆    
	//☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆        

	// (6)学习结果的绘图    
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
	}   //为了显示学习结果，通过输入图像区域的所有像素(特征向量)并进行分类。然后对输入像素用所属等级的颜色绘图。    

	// (7)训练数据的再绘制    
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
	}   //将训练数据在结果图像上重复的绘制出来。    

	// (8)支持向量的绘制    
	sv_num= svm.get_support_vector_count();   
	for (i= 0; i< sv_num; i++) {   
		support = svm.get_support_vector(i);   
		cvCircle(img, cvPoint((int) (support[0] * size), (int) (support[1] * size)), 5, CV_RGB(200, 200, 200));   
	}   
	//用白色的圆圈对支持向量作标记。    

	// (9)图像的显示     
	cvNamedWindow("SVM", CV_WINDOW_AUTOSIZE);   
	cvShowImage("SVM", img);   
	cvWaitKey(0);   
	cvDestroyWindow("SVM");   
	cvReleaseImage(&img);   
	return 0;   
	//显示实际处理结果的图像，直到某个键被按下为止。    
}  
