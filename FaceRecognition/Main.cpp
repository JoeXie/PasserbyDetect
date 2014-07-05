//////////////////////////////////////////////////////////////////////////
// File Name: pjSVM.cpp
// Author:   easyfov(easyfov@gmail.com)
// Company: Lida Optical and Electronic Co.,Ltd.
//http://apps.hi.baidu.com/share/detail/32719017
//////////////////////////////////////////////////////////////////////////

#include <cv.h>
#include <highgui.h>
#include <ml.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <concrt.h>
using namespace std;

#define WIDTH 20
#define HEIGHT 20

int main( int argc, char** argv )
{
	vector<string> img_path;
	vector<int> img_catg;
	int nLine = 0;
	string buf;
	ifstream svm_data( "E:/SVM_DATA.txt" );

	while( svm_data )
	{
		if( getline( svm_data, buf ) )
		{
			nLine ++;
			if( nLine % 2 == 0 )
			{
				img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志（0,1）
			}
			else
			{
				img_path.push_back( buf );//图像路径
			}
		}
	}
	svm_data.close();//关闭文件

	CvMat *data_mat, *res_mat;
	int nImgNum = nLine / 2;			//读入样本数量
	////样本矩阵，nImgNum：横坐标是样本数量， WIDTH * HEIGHT：样本特征向量，即图像大小
	data_mat = cvCreateMat( nImgNum, WIDTH * HEIGHT, CV_32FC1 );
	cvSetZero( data_mat );
	//类型矩阵,存储每个样本的类型标志
	res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );
	cvSetZero( res_mat );

	IplImage *srcImg, *sampleImg;
	float b;
	DWORD n;

	for( string::size_type i = 0; i != img_path.size(); i++ )
	{
		srcImg = cvLoadImage( img_path[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
		if( srcImg == NULL )
		{
			cout<<" can not load the image: "<<img_path[i].c_str()<<endl;
			continue;
		}

		cout<<" processing "<<img_path[i].c_str()<<endl;

		sampleImg = cvCreateImage( cvSize( WIDTH, HEIGHT ), IPL_DEPTH_8U, 1 );//样本大小（WIDTH, HEIGHT）
		cvResize( srcImg, sampleImg );//改变图像大小

		cvSmooth( sampleImg, sampleImg );	//降噪
		//生成训练数据
		n = 0;
		for( int ii = 0; ii < sampleImg->height; ii++ )
		{
			for( int jj = 0; jj < sampleImg->width; jj++, n++ )
			{
				b = (float)((int)((uchar)( sampleImg->imageData + sampleImg->widthStep * ii + jj )) / 255.0 );
				cvmSet( data_mat, (int)i, n, b );
			}
		}
		cvmSet( res_mat, i, 0, img_catg[i] );
		cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;
	}


	CvSVM svm = CvSVM();
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
	param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );
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
	svm.train( data_mat, res_mat, NULL, NULL, param );
	//☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆
	svm.save( "SVM_DATA.xml" );


	//检测样本
	IplImage *tst, *tst_tmp;
	vector<string> img_tst_path;
	ifstream img_tst( "E:/SVM_TEST.txt" );
	while( img_tst )
	{
		if( getline( img_tst, buf ) )
		{
			img_tst_path.push_back( buf );
		}
	}
	img_tst.close();

	CvMat *tst_mat = cvCreateMat( 1, WIDTH*HEIGHT, CV_32FC1 );
	char line[512];
	ofstream predict_txt( "SVM_PREDICT.txt" );
	for( string::size_type j = 0; j != img_tst_path.size(); j++ )
	{
		tst = cvLoadImage( img_tst_path[j].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
		if( tst == NULL )
		{
			cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;
			continue;
		}
		tst_tmp = cvCreateImage( cvSize( WIDTH, HEIGHT ), IPL_DEPTH_8U, 1 );
		cvResize( tst, tst_tmp );
		cvSmooth( tst_tmp, tst_tmp );
		n = 0;
		for(int ii = 0; ii < tst_tmp->height; ii++ )
		{
			for(int jj = 0; jj < tst_tmp->width; jj++, n++ )
			{
				b = (float)(((int)((uchar)tst_tmp->imageData+tst_tmp->widthStep*ii+jj))/255.0);
				cvmSet( tst_mat, 0, n, (double)b );
			}
		}

		int ret = svm.predict( tst_mat );
		sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );
		predict_txt<<line;
	}
	predict_txt.close();

	cvReleaseImage( &srcImg );
	cvReleaseImage( &sampleImg );
	cvReleaseImage( &tst );
	cvReleaseImage( &tst_tmp );
	cvReleaseMat( &data_mat );
	cvReleaseMat( &res_mat );

	return 0;
}

