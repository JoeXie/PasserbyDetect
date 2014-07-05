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
				img_catg.push_back( atoi( buf.c_str() ) );//atoi���ַ���ת�������ͣ���־��0,1��
			}
			else
			{
				img_path.push_back( buf );//ͼ��·��
			}
		}
	}
	svm_data.close();//�ر��ļ�

	CvMat *data_mat, *res_mat;
	int nImgNum = nLine / 2;			//������������
	////��������nImgNum�������������������� WIDTH * HEIGHT������������������ͼ���С
	data_mat = cvCreateMat( nImgNum, WIDTH * HEIGHT, CV_32FC1 );
	cvSetZero( data_mat );
	//���;���,�洢ÿ�����������ͱ�־
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

		sampleImg = cvCreateImage( cvSize( WIDTH, HEIGHT ), IPL_DEPTH_8U, 1 );//������С��WIDTH, HEIGHT��
		cvResize( srcImg, sampleImg );//�ı�ͼ���С

		cvSmooth( sampleImg, sampleImg );	//����
		//����ѵ������
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
	svm.train( data_mat, res_mat, NULL, NULL, param );
	//�������ѵ�����ݺ�ȷ����ѧϰ����,����SVMѧϰ�����
	svm.save( "SVM_DATA.xml" );


	//�������
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

