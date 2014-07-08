#include <opencv2/opencv.hpp>

#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

//��������
int Train();
int train(char* positivePath, int positiveSampleCount, 
	char* negativePath, int negativeSampleCount, char* classifierSavePath);
void pictureDetect(Mat img, char* svmDetectorPath);
void VideoDetect();

int main(int argc, char* argv[])
{
	if(  0  ) //����Ҫ��Ҫѵ��
	{
		cout << "��ʼѵ��" << endl;
		int trainFlag = train("E:\\SmartCity\\������\\������_������һ����_13\\", 865, 
			"E:\\SmartCity\\������\\������13(�Զ���)\\64x128\\", 775, "E:\\SmartCity\\Result\\Result13\\");
	}

	if(  1  ) //����Ҫ��Ҫ���
	{
		Mat img;
		//img = imread("E:\\�ǻ۳���\\�ǻ۳���WD\\��֤����\\1_2_01_1\\hongsilounorth_13_1920x1080_30_R1\\0005651.jpg");
		img = imread("e:/smartcity/pic.jpg");
		if(img.data == NULL)
		{
			printf("û��ͼƬ\n");
			system("pause");
			return -1;
		}
		pictureDetect(img, "E:\\SmartCity\\Result\\12472+856_RBF\\SVMDetector.txt");
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

		cv::Mat img = cv::imread(positiveImagePath);
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

	//��ʼѵ��SVM
	CvSVMParams params;  
	params.svm_type = CvSVM::C_SVC;  
	params.kernel_type = CvSVM::RBF;  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	params.C = 0.01;   // Ĭ��ʱC��1

	Mysvm svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, 0, 0, params ); //��SVM������ѵ��
	// train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, 
	// const CvMat* sampleIdx=0, CvSVMParams params=CvSVMParams() )
	char* trainSavePath = new char[200];
	strcpy(trainSavePath, classifierSavePath);
	strcat(trainSavePath, "ResultOfTrain.txt");
	svm.save(trainSavePath);  //�洢ѵ�����
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

	double* alphaArr = svm.get_alpha();   //��������һ��,�ѵ���͸���ȣ�
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
	fprintf(fp, "%f", rho);
	cout<< SVMDetectorSavePath <<"֧�������������"<<endl;//����֧������

	delete [] SVMDetectorSavePath;
	fclose(fp);

	time_t endTime = time(NULL);	//��¼����ʱ��
	cout <<"ѵ������ʱ�䣺" << difftime(endTime, startTime) << "��" << endl; //��ӡѵ����ʱ

	return 1;
}

void pictureDetect(Mat img, char* svmDetectorPath)
{	
	time_t startTime = time(NULL);	//��¼��ʼʱ��

	//���ļ��е����ݶ��뵽x��
	vector<float> x;
	ifstream fileIn(svmDetectorPath, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), 
	// Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
	// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	hog.setSVMDetector(x);//ѵ���ķ�����
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
		for(int i = 0; i < found.size(); i++)
		{
			rectangle(img, found[i], Scalar(0, 0, 255), 2); //��ͼ���ϻ�������
		}
	}

	time_t endTime = time(NULL);	//��¼����ʱ��
	cout <<"�������ʱ�䣺" << difftime(endTime, startTime) << "��" << endl; //��ӡѵ����ʱ

	cvNamedWindow("�������", CV_WINDOW_NORMAL);
	imshow("�������", img);
	cout << "�����ɣ�����ʾ�����" << endl;

//	cvSaveImage("e:/SmartCity/ProcessedImage.jpg", &img, NULL);
	waitKey(0);
}

class Mysvm: public CvSVM
{
public:
	int get_alpha_count()
	{
		return this->sv_total;
	}

	int get_sv_dim()
	{
		return this->var_all;
	}

	int get_sv_count()
	{
		return this->decision_func->sv_count;
	}

	double* get_alpha()
	{
		return this->decision_func->alpha;
	}

	float** get_sv()
	{
		return this->sv;
	}

	float get_rho()
	{
		return this->decision_func->rho;
	}
};