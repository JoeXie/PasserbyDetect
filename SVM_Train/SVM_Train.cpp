#include <opencv2/opencv.hpp>
#include "Mysvm.h"
#include "MyRect.h"

#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

//��������
int train(char* positivePath, int positiveSampleCount, 
	char* negativePath, int negativeSampleCount, char* classifierSavePath);
void pictureDetect(IplImage* img, char* svmDetectorPath);
void saveDetectResult(IplImage* img, CvRect rect, char* savePath);



int main(int argc, char* argv[])
{
	if( 0 ) //����Ҫ��Ҫѵ��
	{
		cout << "��ʼѵ��" << endl;
		int trainFlag = train("E:\\SmartCity\\������\\Pos_Mixed\\", 290, 
			"E:\\SmartCity\\������\\Neg_Auto_13\\64x128\\", 424, "E:\\SmartCity\\Result\\Result_13\\");
	}

	if( 0 ) //����Ҫ��Ҫ���
	{
		IplImage* img = NULL;
		
		img = cvLoadImage("E:\\SmartCity\\���ݼ�\\��֤����\\1_2_01_1\\hongsilounorth_13_1920x1080_30_R1\\0005926.jpg");
		//img = cvLoadImage("E:\\SmartCity\\004.jpg");
		if(img == NULL)
		{
			printf("û��ͼƬ\n");
			system("pause");
			return -1;
		}
		pictureDetect(img, "E:\\SmartCity\\Result\\Result_13\\SVMDetector.txt");
	}

	system("pause");
	return 1;
}

int train(char* positivePath, int positiveSampleCount, 
	char* negativePath, int negativeSampleCount, char* classifierSavePath)
{
	time_t startTime = time(NULL);	//��¼��ʼʱ��

	int totalSampleCount = positiveSampleCount + negativeSampleCount; //�ܵ�������
	cout<<"//////////////////////////////////////////////////////////////////"<<endl;
	cout<<"totalSampleCount: "<<totalSampleCount<<endl;
	cout<<"positiveSampleCount: "<<positiveSampleCount<<endl;
	cout<<"negativeSampleCount: "<<negativeSampleCount<<endl;

	CvMat *sampleFeaturesMat = cvCreateMat(totalSampleCount , 3780, CV_32FC1);
	//64*128��ѵ���������þ�����totalSample*3780,64*64��ѵ���������þ�����totalSample*1764
	cvSetZero(sampleFeaturesMat);  
	CvMat *sampleLabelMat = cvCreateMat(totalSampleCount, 1, CV_32FC1);//������ʶ  
	cvSetZero(sampleLabelMat);  


	cout<<"************************************************************"<<endl;
	cout<<"start to training positive samples."<<endl;

	char* positiveNamelistPath = new char[200]; //�ı��ĵ������ļ�·��
	strcpy(positiveNamelistPath, positivePath);
	strcat(positiveNamelistPath, "Namelist.txt"); //����ͼ���ļ����ı��ĵ�������·����

	FILE* fileIn = fopen(positiveNamelistPath, "r");  //���ļ������룬���ط���ָ��FILE�����ָ��
	if(fileIn == NULL)
	{
		printf("Can not open file."); 
		return 0;
	}

	delete[] positiveNamelistPath; //�ͷ��ڴ�

	for(int i=0; i<positiveSampleCount; i++)  
	{  
		char* positiveImageName = new char[100]; //ͼ���ļ���
		fscanf(fileIn,"%s",positiveImageName); //��ȡ�ļ�������whitespaceֹͣ
		char* positiveImagePath = new char[200]; //�洢�����ļ�·��
		strcpy(positiveImagePath, positivePath);
		strcat(positiveImagePath,positiveImageName); //���ڶ����ַ���ƴ�ӵ���һ������
		printf("%s\n",positiveImagePath); //��ӡ�������ļ�·����

		Mat img = cv::imread(positiveImagePath);
		if(img.data == NULL)
		{
			cout<<"negative image sample load error: "<<positiveImagePath<<endl;
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
		// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), 
		// Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, 
		// double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
		// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
		vector<float> featureVec; //����һ��Vector����

		hog.compute(img, featureVec, cv::Size(8,8));  //����HOG����
		int featureVecSize = featureVec.size();

		for (int j=0; j<featureVecSize; j++)  
		{  		
			CV_MAT_ELEM( *sampleFeaturesMat, float, i, j ) = featureVec[j]; //���������Ƶ�sampleFeaturesMat�ĵ�i��
		
		}  

		sampleLabelMat->data.fl[i] = 1;

		delete [] positiveImageName;
		delete [] positiveImagePath;

	}
	fclose(fileIn);

	cout<<"end of training for positive samples..."<<endl;

	cout<<"*********************************************************"<<endl;
	cout<<"start to train negative samples..."<<endl;

	char* negativeNamelistPath = new char[200]; //�ı��ĵ������ļ�·��
	strcpy(negativeNamelistPath, negativePath);
	strcat(negativeNamelistPath, "Namelist.txt"); //����ͼ���ļ����ı��ĵ�������·����
	// FILE* fileIn;
	if((fileIn=fopen(negativeNamelistPath, "r"))==NULL) //���ļ������룬���ط���ָ��FILE�����ָ��
	{
		printf("Can not open file.");
		return 0;
	}

	delete [] negativeNamelistPath;
	
	for (int i=0; i<negativeSampleCount; i++)
	{  
		char* negativeImageName = new char[100]; //ͼ���ļ���
		fscanf(fileIn,"%s",negativeImageName); //��ȡ�ļ�������whitespaceֹͣ
		char* negativeImagePath = new char[200]; //�洢�����ļ�·��
		strcpy(negativeImagePath, negativePath);
		strcat(negativeImagePath,negativeImageName); //���ڶ����ַ���ƴ�ӵ���һ������
		printf("%s\n",negativeImagePath); //��ӡ�������ļ�·����
		
		cv::Mat img = cv::imread(negativeImagePath);
		if(img.data == NULL)
		{
			cout<<"negative image sample load error: "<<negativeImagePath<<endl;
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);  
		vector<float> featureVec; 

		hog.compute(img,featureVec,cv::Size(8,8));//����HOG����
		int featureVecSize = featureVec.size();  

		for ( int j=0; j<featureVecSize; j ++)  
		{  
			CV_MAT_ELEM( *sampleFeaturesMat, float, i + positiveSampleCount, j ) = featureVec[ j ];
		}  

		sampleLabelMat->data.fl[ i + positiveSampleCount ] = -1;

		delete [] negativeImageName;
		delete [] negativeImagePath;
	}  
	fclose(fileIn);

	cout<<"end of training for negative samples."<<endl;
	cout<<"********************************************************"<<endl;
	cout<<"start to train for SVM classifier."<<endl;
	
	//����ѵ������
	//CvSVMParams params;  
	//params.svm_type = CvSVM::C_SVC;  
	//params.kernel_type = CvSVM::LINEAR;  
	//params.C = 0.01;   // Ĭ��ʱC��1
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON); 
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
	CvSVMParams params(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
	cout << "ѵ��������" << params.svm_type <<", "<< params.kernel_type<< endl;

	//��ʼѵ��SVM
	Mysvm svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, 0, 0, params ); //��SVM������ѵ��
	// train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, 
	// const CvMat* sampleIdx=0, CvSVMParams params=CvSVMParams() )
	char* trainSavePath = new char[200];
	strcpy(trainSavePath, classifierSavePath);
	strcat(trainSavePath, "SVM_HOG.xml");
	svm.save(trainSavePath, 0);  //�洢ѵ�����
	cout<< trainSavePath <<"ѵ������������" << endl;

	delete [] trainSavePath;

	cvReleaseMat(&sampleFeaturesMat);
	cvReleaseMat(&sampleLabelMat);

	int supportVectorCount = svm.get_support_vector_count();
	cout<<"support vector size of SVM��"<<supportVectorCount<<endl;
	cout<<"************************ end of training for SVM ******************"<<endl;

	CvMat *sv,*alp,*re;//���������������� 
	sv  = cvCreateMat(supportVectorCount , 3780, CV_32FC1);
	alp = cvCreateMat(1 , supportVectorCount, CV_32FC1);
	re  = cvCreateMat(1 , 3780, CV_32FC1);
	CvMat *res  = cvCreateMat(1 , 1, CV_32FC1);

	cvSetZero(sv);
	cvSetZero(re);
  
	for(int i=0; i<supportVectorCount; i++)
	{
		memcpy( (float*)(sv->data.fl+i*3780), svm.get_support_vector(i), 3780*sizeof(float));	//������������
	}

	double* alphaArr = svm.get_alpha();   
	int alphaCount = svm.get_alpha_count();

	for(int i=0; i<supportVectorCount; i++)
	{
        alp->data.fl[i] = alphaArr[i];
	}
	cvMatMul(alp, sv, re);	//re = alp*sv

	int posCount = 0;
	for (int i=0; i<3780; i++)
	{
		re->data.fl[i] *= -1;
	}

	//��re�е�����д��text�ĵ���
	char* SVMDetectorSavePath = new char[200];
	strcpy(SVMDetectorSavePath, classifierSavePath);
	strcat(SVMDetectorSavePath, "SVMDetector.txt");
	FILE* fp = fopen(SVMDetectorSavePath,"wb");
	if( NULL == fp )
	{
		return 1;
	}
	for(int i=0; i<3780; i++)
	{
		fprintf(fp,"%f \n",re->data.fl[i]);
	}

	float rho = svm.get_rho();
	fprintf(fp, "%f", rho); //���������
	cout<< SVMDetectorSavePath <<"֧�������������"<<endl;

	delete [] SVMDetectorSavePath;
	fclose(fp);

	time_t endTime = time(NULL);	//��¼����ʱ��
	cout <<"ѵ������ʱ�䣺" << difftime(endTime, startTime) << "��" << endl; //��ӡѵ����ʱ

	return 1;
}

void pictureDetect(IplImage* img, char* svmDetectorPath)
{	
	time_t startTime = time(NULL);	//��¼��ʼʱ��

	//���ļ��е����ݶ��뵽x��
	vector<float> x;
	ifstream fileIn(svmDetectorPath, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn >> val;
		x.push_back(val);
	}
	fileIn.close();
	
	if (x.size() == 3781)
		cout << svmDetectorPath << "����������ɹ���" << endl;
	else
	{
		cout << "�������������" <<endl;
		return;
	}

	cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), 
	// Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
	// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	hog.setSVMDetector(x); //ѵ���ķ�����
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//�Դ��ķ�����

	//cvNamedWindow("img", 0);

	cout << "���ڼ��ͼ��" << endl;
	vector<cv::Rect>  found; //�洢�����
	hog.detectMultiScale(img, found, 0.0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
	// detectMultiScale(const GpuMat& img, vector<Rect>& found_locations, double hit_threshold=0, 
	// Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)

	if (found.size() > 0) 
	{
		cout << "���ο���Ŀ��"<< found.size() << endl;

		// ����������ͼ
		for(int i = 0; i < found.size(); i++)
		{
			if((found[i].x >= 0) && ((found[i].x + found[i].width) <= img->width) 
				&& (found[i].y >= 0) && (found[i].y + found[i].height <= img->height))
			{
				char* name = new char[100]; //�ļ���
				char* prePath = new char[200]; // �ļ���·��
				itoa(i, name, 10);
				strcat(name, ".png"); //�����ļ��� i.png
				strcpy(prePath, "E:\\SmartCity\\Result\\SaveDetectResult\\"); 
				strcat(prePath, name); //�γ�����·��

				cout << prePath << endl;

				/*IplImage * image = cvCreateImageHeader(cvSize(img.cols, img.rows), 8, 3);
				image = cvGetImage(&img, image);*/
				saveDetectResult(img, found[i], prePath); //�����ͼ
				cout << prePath << " ������ϡ�" << endl;

				delete[] name;
				delete[] prePath;
			}
		}

		//ȥǶ��
		Vector<Rect> found_NoNest;
		int ii = 0;
		for(int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			//��������for������ҳ�����û��Ƕ�׵ľ��ο�r,������found_NoNest��,�����Ƕ�׵�
			//��,��ȡ���������Ǹ����ο����found_NoNest��
            for(ii = 0; ii <found.size(); ii++)
                if(ii != i && (r&found[ii])==r)
                    break;
            if(ii == found.size())
               found_NoNest.push_back(r);
		}

		//ȥ�ص�
		Vector<Rect> found_NoOverlap;
		for (int i = 0; i < found_NoNest.size(); i++)
		{
			
		}
	

		 

		//��ͼ���ϻ�������

		for(int i = 0; i < found.size(); i++)
		{
			


			cvRectangle(img, found[i].tl(), found[i].br(), Scalar(0, 255, 0), 3); 
			


			/*Rect r = found[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);*/
		}
	}

	time_t endTime = time(NULL);	//��¼����ʱ��
	cout <<"�������ʱ�䣺" << difftime(endTime, startTime) << "��" << endl; //��ӡѵ����ʱ

	cvNamedWindow("�������", CV_WINDOW_NORMAL);
	cvShowImage("�������", img);
	cout << "�����ɣ�����ʾ�����" << endl;

//	cvSaveImage("e:/SmartCity/ProcessedImage.jpg", &img, NULL);
	waitKey(0);
}

void saveDetectResult(IplImage* img, CvRect rect, char* savePath)
{
	CvMat *subMat = cvCreateMatHeader(rect.width, rect.height, CV_8UC3); //����һ��rect.width * rect.height�ľ���ͷ

	subMat = cvGetSubRect(img, subMat, rect); //pImgΪָ��ͼ���ָ�룬subMatָ��洢����ͼ��ľ��󣬷���ֵ��subMat���

	IplImage *subImg = cvCreateImageHeader(cvSize(rect.width, rect.height), 8, 3); //����һ��rect.width * rect.height��ͼ��ͷ
	cvGetImage(subMat, subImg); //subMatΪ�洢���ݵľ���SubImgָ��ͼ�񣬷���ֵ��SubImg���

	cvSaveImage(savePath, subImg, 0);
}
