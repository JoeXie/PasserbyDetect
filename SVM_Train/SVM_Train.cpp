#include <opencv2/opencv.hpp>

#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

//函数声明
int Train();
void PictureDetect(Mat img);
void VideoDetect();

int main(int argc, char* argv[])
{
	time_t startTime = time(NULL);	//记录开始时间

	int trainFlag = Train();
	
	time_t endTime = time(NULL);	//记录结束时间

	cout <<"训练所用时间：" << difftime(endTime, startTime) << "秒" << endl; //打印训练用时

	if(trainFlag = 1)
		cout<< "Train Successed!" << endl;
	else
	{
		cout<< "Train Fail" << endl;
		return -1;
	};

	Mat img;
	img = imread(argv[1]);

	if(argc != 2 || !img.data){
		printf("没有图片\n");
		return -1;
	}
	PictureDetect(img);
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

int Train()
{
	char classifierSavePath[256] = "e:/SmartCity/pedestrianDetect-peopleFlow.txt"; //存储SVM训练结果

	string positivePath = "E:\\SmartCity\\PositiveImages\\";  //正样本路径
	string negativePath = "E:\\SmartCity\\NegativeImages\\";

	int positiveSampleCount = 4037;  //正样本数目
	int negativeSampleCount = 117;
	int totalSampleCount = positiveSampleCount + negativeSampleCount;

	cout<<"//////////////////////////////////////////////////////////////////"<<endl;
	cout<<"totalSampleCount: "<<totalSampleCount<<endl;
	cout<<"positiveSampleCount: "<<positiveSampleCount<<endl;
	cout<<"negativeSampleCount: "<<negativeSampleCount<<endl;

	CvMat *sampleFeaturesMat = cvCreateMat(totalSampleCount , 3780, CV_32FC1);
	//64*128的训练样本，该矩阵将是totalSample*3780,64*64的训练样本，该矩阵将是totalSample*1764
	cvSetZero(sampleFeaturesMat);  
	CvMat *sampleLabelMat = cvCreateMat(totalSampleCount, 1, CV_32FC1);//样本标识  
	cvSetZero(sampleLabelMat);  

	cout<<"************************************************************"<<endl;
	cout<<"start to training positive samples..."<<endl;

	char positiveImgName[256];
	string path;
	for(int i=0; i<positiveSampleCount; i++)  
	{  
		memset(positiveImgName, '\0', 256*sizeof(char));	//Initializes or sets device memory to a value.
		sprintf(positiveImgName, "%d.jpg", i);
		int len = strlen(positiveImgName);
		string tempStr = positiveImgName;
		path = positivePath + tempStr;

		cv::Mat img = cv::imread(path);
		if( img.data == NULL )
		{
			cout<<"positive image sample load error: "<<i<<" "<<path<<endl;
			system("pause");
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
		// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), 
		// Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, 
		// double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
		// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
		vector<float> featureVec; //定义一个Vector容器

		hog.compute(img, featureVec, cv::Size(8,8));  //计算HOG特征
		int featureVecSize = featureVec.size();

		for (int j=0; j<featureVecSize; j++)  
		{  		
			CV_MAT_ELEM( *sampleFeaturesMat, float, i, j ) = featureVec[j]; //把特征复制到sampleFeaturesMat的第i行
			/*if(i == 2000)
			{
				system("pause");
			}*/
		}  
//		cout << i <<","<< endl;
		sampleLabelMat->data.fl[i] = 1;
	}

	cout<<"end of training for positive samples..."<<endl;

	cout<<"*********************************************************"<<endl;
	cout<<"start to train negative samples..."<<endl;

	//计算负样本HOG
	char negativeImgName[256];
	for (int i=0; i<negativeSampleCount; i++)
	{  
		memset(negativeImgName, '\0', 256*sizeof(char)); //内存清零
		sprintf(negativeImgName, "%d.jpg", i);
		path = negativePath + negativeImgName;
		cv::Mat img = cv::imread(path);
		if(img.data == NULL)
		{
			cout<<"negative image sample load error: "<<path<<endl;
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);  
		vector<float> featureVec; 

		hog.compute(img,featureVec,cv::Size(8,8));//计算HOG特征
		int featureVecSize = featureVec.size();  

		for ( int j=0; j<featureVecSize; j ++)  
		{  
			CV_MAT_ELEM( *sampleFeaturesMat, float, i + positiveSampleCount, j ) = featureVec[ j ];
		}  

		sampleLabelMat->data.fl[ i + positiveSampleCount ] = -1;
	}  

	cout<<"end of training for negative samples..."<<endl;
	cout<<"********************************************************"<<endl;
	cout<<"start to train for SVM classifier..."<<endl;

	//开始训练SVM
	CvSVMParams params;  
	params.svm_type = CvSVM::C_SVC;  
	params.kernel_type = CvSVM::RBF;  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	params.C = 0.01;   // 默认时C＝1

	Mysvm svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, 0, 0, params ); //用SVM分类器训练
	// train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, 
	// const CvMat* sampleIdx=0, CvSVMParams params=CvSVMParams() )
	svm.save(classifierSavePath);  //存储训练结果

	cvReleaseMat(&sampleFeaturesMat);
	cvReleaseMat(&sampleLabelMat);

	int supportVectorCount = svm.get_support_vector_count();
	cout<<"support vector size of SVM："<<supportVectorCount<<endl;
	cout<<"************************ end of training for SVM ******************"<<endl;

	CvMat *sv,*alp,*re;//所有样本特征向量 
	sv  = cvCreateMat(supportVectorCount , 3780, CV_32FC1);
	alp = cvCreateMat(1 , supportVectorCount, CV_32FC1);
	re  = cvCreateMat(1 , 3780, CV_32FC1);
	CvMat *res  = cvCreateMat(1 , 1, CV_32FC1);

	cvSetZero(sv);
	cvSetZero(re);
  
	for(int i=0; i<supportVectorCount; i++)
	{
		memcpy( (float*)(sv->data.fl+i*3780), svm.get_support_vector(i), 3780*sizeof(float));	//复制特征向量
	}

	double* alphaArr = svm.get_alpha();   //看不懂这一段,难道是透明度？
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

	//把re中的数据写到text文档里
	FILE* fp = fopen("e:/SmartCity/hogSVMDetector-peopleFlow.txt","wb");
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
	cout<<"e:/SmartCity/hogSVMDetector.txt 保存完毕"<<endl;//保存HOG能识别的分类器
	fclose(fp);

	return 1;
}

void PictureDetect(Mat img)
{	
	//将文件中的数据读入到x中
	vector<float> x;
	ifstream fileIn("e:/SmartCity/hogSVMDetector-peopleFlow.txt", ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	vector<cv::Rect>  found;
	cv::HOGDescriptor hog(cv::Size(64,128), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), 
	// Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
	// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	hog.setSVMDetector(x);//训练的分类器
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//自带的分类器

	//cvNamedWindow("img", 0);

	hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
	// detectMultiScale(const GpuMat& img, vector<Rect>& found_locations, double hit_threshold=0, 
	// Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)
	if (found.size() > 0)
	{
		//for (int i=0; i<found.size(); i++)
		//{
		//	CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
		//	cvRectangle(img, cvPoint(tempRect.x,tempRect.y), 
		//		cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);
		//	//在图像上画出矩形
		//}
		for(int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			rectangle(img, r.tl(), r.br(), Scalar(0, 0, 255), 3);
		}
	}
	cvNamedWindow("检测行人", CV_WINDOW_NORMAL);
	imshow("检测行人", img);

	waitKey(0);
}

void VideoDetect()
{
	CvCapture* cap = cvCreateFileCapture("E:\\02.avi");
	if (!cap)
	{
		cout<<"avi file load error..."<<endl;
		system("pause");
		exit(-1);
	}

	//将文件中的数据读入到x中
	vector<float> x;
	ifstream fileIn("e:/SmartCity/hogSVMDetector-peopleFlow.txt", ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();


	vector<cv::Rect>  found;
	cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), 
	// Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
	// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	hog.setSVMDetector(x);

	IplImage* img = NULL;
	cvNamedWindow("img", 0);
	while(img=cvQueryFrame(cap))
	{
		hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
		// detectMultiScale(const GpuMat& img, vector<Rect>& found_locations, double hit_threshold=0, 
		// Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)
		if (found.size() > 0)
		{

			for (int i=0; i<found.size(); i++)
			{
				CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
				cvRectangle(img, cvPoint(tempRect.x,tempRect.y), 
					cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);
				//在图像上画出矩形
			}
		}
	}
	cvReleaseCapture(&cap);
}