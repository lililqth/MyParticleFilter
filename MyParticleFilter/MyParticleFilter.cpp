#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>

using namespace std;
using namespace cv;
#define B(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3]		//B
#define G(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+1]	//G
#define R(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+2]	//R
#define ALPHA_COEFFICIENT      0.3													// 目标模型更新权重取0.1-0.3

const int G_BIN = 8;
const int B_BIN = 8;
const int R_BIN = 8;
const int R_SHIFT = 5;
const int B_SHIFT = 5;
const int G_SHIFT = 5;

const int H_BIN = 10;
const int S_BIN = 9;

const int NParticle = 100;				// 粒子的个数
const int Num = 10;						//帧差的间隔
const int T = 40;						//Tf
const int Re = 30;						//
const int ai = 0.08;					//学习率
long ran_seed = 802163120; 				//随机数种子
typedef struct SpaceState 				//空间状态
{
	int xt;               				// x坐标位置
	int yt;               				// x坐标位置
	float v_xt;           				// x方向运动速度
	float v_yt;           				// y方向运动速度
	int Hxt;              				// x方向半窗宽
	int Hyt;              				// y方向半窗宽
	float at_dot;         				// 尺度变换速度
}SPACESTATE;

IplImage *curFrame = NULL;				// 当前帧
unsigned char * img;					//把iplimg改到char*  便于计算
int Wid, Hei;							//图像的大小
int WidIn, HeiIn;						//输入的半宽和半高
int xIn, yIn;							//跟踪时输入的中心点
int xOut, yOut;							//跟踪时输出的中心点
int widIn,heiIn;						//输入的半宽与半高
int widOut,heiOut;						//输出的半宽与半高
bool bSelectObject = false;
Point origin;
Rect selection;
bool pause = false; 					//是否暂停
bool track = false; 					//是否跟踪

SPACESTATE *states = NULL;	  			//状态数组
float *weights = NULL;					//每个粒子的权重
int nbin;					    		// 直方图条数
float *modelHist = NULL;				// 存放模型的直方图
float Pi_Thres = (float)0.90;			// 权重阈值
float Weight_Thres = (float)0.0001;		// 最大权重阈值，用来判断是否目标丢失

const float DELTA_T = (float)0.05;		//帧频，可以为30，25，15，10等
const float VELOCITY_DISTURB = 40.0;	//速度扰动幅值
const float SCALE_DISTURB = 0.0;		//窗口宽高扰动幅度
const float SCALE_CHANGE_D = 0.001;		//尺度变换速度扰动幅值
void clearAll()
{
	if(states != NULL)
	{
		delete [] states;
	}
	if(weights != NULL)
	{
		delete [] weights;
	}
	if(modelHist != NULL)
	{
		delete [] modelHist;
	}
	delete [] img;
}

void Rgb2Hsv(float R, float G, float B, float& H, float& S, float&V)
{
     // r,g,b values are from 0 to 1
    // h = [0,360], s = [0,1], v = [0,1]
    // if s == 0, then h = -1 (undefined)
    float min, max, delta,tmp;
    tmp = R>G?G:R;
    min = tmp>B?B:tmp;
    tmp = R>G?R:G;
    max = tmp>B?tmp:B;
    V = max; // v
    delta = max - min;
    if(max != 0)
        S = delta / max; // s
    else
    {
        // r = g = b = 0 // s = 0, v is undefined
        S = 0;
        H = 0;
        return;
    }
    if (delta == 0){
        H = 0;
        return;
    }
    else if(R == max){
        if (G >= B)
            H = (G - B) / delta; // between yellow & magenta
        else
            H = (G - B) / delta + 6.0;
    }
    else if(G == max)
        H = 2.0 + (B - R) / delta; // between cyan & yellow
    else if (B == max)
        H = 4.0 + (R - G) / delta; // between magenta & cyan
    H *= 60.0; // degrees
}

void calcuColorHistogram(int x0, int y0, int Wx, int Hy,
						 unsigned char * image, int W, int H,
						 float * ColorHist, int bins)
{
	int xBegin, yBegin;//图像区域的左上角坐标
	int xEnd, yEnd;

	for (int i=0; i<bins; i++)
	{
		ColorHist[i] = 0.0;
	}
	if ((x0 < 0) || (x0 >= W) || (y0 < 0) || (y0 >= H)
		|| (Wx <= 0) || (Hy <= 0)) return;
	xBegin = (x0 - Wx) < 0.0 ? 0 : (x0 - Wx);
	yBegin = (y0 - Hy) < 0.0 ? 0 : (y0 - Hy);
	xEnd = (x0 + Wx) >= W ? (W - 1) : (x0 + Wx);
	yEnd = (y0 + Hy) >= H ? (H - 1) : (y0 + Hy);
	int a2 = Wx*Wx+Hy*Hy;                // 计算核函数半径平方a^2
	float f = 0.0;
	int r,g,b;
	for (int j = yBegin; j<=yEnd; j++)
	{
		for (int i = xBegin; i <= xEnd; i++)
		{
			r = image[(j * W + i) * 3] >> R_SHIFT;
			g = image[(j * W + i) * 3 + 1] >> G_SHIFT;
			b = image[(j * W + i) * 3 + 2] >> B_SHIFT;
		/*	r = image[(j * W + i) * 3];
			g = image[(j * W + i) * 3 + 1];
			b = image[(j * W + i) * 3 + 2];
			float H, S, V;
			Rgb2Hsv(r, g, b, H, S, V);
			H = (int)(H / 40);
			S = (int)(S * 8);*/
			int index = r*G_BIN * B_BIN + g * B_BIN + b;
			//int index = H * S_BIN + S;
			float r2 = (float)(((j-y0)*(j-y0)+(i-x0)*(i-x0))*1.0/a2);
			float k = 1 - r2;
			f = f + k;
			ColorHist[index] =ColorHist[index] + k;
		}
	}
	for (int i = 0; i < bins; i++)
	{
		ColorHist[i] = ColorHist[i]/f;
	}
}



float randGaussian(float u, float sigma)
{
	float x1, x2, v1, v2;
	float s = 100.0;
	float y;

	/*
	使用筛选法产生正态分布N(0,1)的随机数(Box-Mulles方法)
	1. 产生[0,1]上均匀随机变量X1,X2
	2. 计算V1=2*X1-1,V2=2*X2-1,s=V1^2+V2^2
	3. 若s<=1,转向步骤4，否则转1
	4. 计算A=(-2ln(s)/s)^(1/2),y1=V1*A, y2=V2*A
	y1,y2为N(0,1)随机变量
	*/
	while (s >= 1.0 || s == 0)
	{
		x1 = (float)rand();
		x2 = (float)rand();
		x1 = x1 / RAND_MAX;
		x2 = x2 / RAND_MAX;
		v1 = 2 * x1 - 1;
		v2 = 2 * x2 - 1;
		s = v1*v1 + v2*v2;
	}
	y = (float)(sqrt(-2.0 * log(s)/s) * v1);
	/*
	根据公式
	z = sigma * y + u
	将y变量转换成N(u,sigma)分布
	*/
	return(sigma * y + u);
}

void propagate(SPACESTATE * state, int N)
{
	int i;
	int j;
	float rn[7];

	// 对每一个状态向量state[i](共N个)进行更新
	for (i = 0; i < N; i++)  // 加入均值为0的随机高斯噪声
	{
		for (j = 0; j < 7; j++) rn[j] = randGaussian(0, (float)0.6); /* 产生7个随机高斯分布的数 */
		state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);
		state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
		state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
		state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
		state[i].Hxt = (int)(state[i].Hxt+state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
		state[i].Hyt = (int)(state[i].Hyt+state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
		state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
		cvCircle(curFrame, Point(state[i].xt,state[i].yt) ,3 , CV_RGB(0,255,0),1, 8, 3);
	}
	return;
}

int initialize(int x0, int y0, int Wx, int Hy,
			   unsigned char *img, int W, int H)
{
	float rn[7];
	srand(unsigned(time(0)));
	states = new SPACESTATE [NParticle];
	assert(states != NULL);
	weights = new float [NParticle];
	assert(weights != NULL);
	nbin = R_BIN * G_BIN * B_BIN;//确定直方图条数
//	nbin = H_BIN * S_BIN;
	modelHist = new float[nbin];//申请直方图内存
	assert(modelHist != NULL);

	//计算目标模板直方图
	calcuColorHistogram(x0, y0, Wx, Hy, img, W, H, modelHist, nbin);

	//初始化粒子状态(以(x0,y0,1,1,Wx,Hy,0.1)为中心呈N(0,0.4)正态分布)
	states[0].xt = x0;
	states[0].yt = y0;
	states[0].v_xt = (float)0.0; 		// 1.0
	states[0].v_yt = (float)0.0; 		// 1.0
	states[0].Hxt = Wx;
	states[0].Hyt = Hy;
	states[0].at_dot = (float)0.0;		// 0.1
	weights[0] = (float)(1.0/NParticle);// 0.9;

	// 初始化粒子状态(以(x0,y0,1,1,Wx,Hy,0.1)为中心呈N(0,0.4)正态分布)
	for (int i = 1; i < NParticle; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			rn[j] = randGaussian(0, (float)0.6);
		}
		states[i].xt = (int)(states[0].xt + rn[0] * Wx);
		states[i].yt = (int)(states[0].yt + rn[1] * Hy);
		states[i].v_xt = (float)(states[0].v_xt + rn[2] * VELOCITY_DISTURB);
		states[i].v_yt = (float)(states[0].v_yt + rn[3] * VELOCITY_DISTURB);
		states[i].Hxt = (int)(states[0].Hxt + rn[4] * SCALE_DISTURB);
		states[i].Hyt = (int)(states[0].Hyt + rn[5] * SCALE_DISTURB);
		states[i].at_dot = (float)(states[0].at_dot + rn[6] * SCALE_CHANGE_D);
		cvCircle(curFrame, Point(states[i].xt,states[i].yt) ,3 , CV_RGB(0,255,0),1, 8, 3);
		// 权重统一为1/N，让每个粒子有相等的机会
		weights[i] = (float)(1.0/NParticle);
	}
	return 1;
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{
	int centerX, centerY;
	if (bSelectObject)
	{
		selection.x = MIN(origin.x, x);
		selection.y = MIN(origin.y, y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
	}
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		bSelectObject = true;
		track = false;
		pause = true;
		break;
	case CV_EVENT_LBUTTONUP:
		bSelectObject = false;
		centerX = selection.x + selection.width / 2;
		centerY = selection.y + selection.height / 2;
		WidIn = selection.width / 2;
		HeiIn = selection.height / 2;
		track = true;
		pause = false;
		initialize(centerX, centerY, WidIn, HeiIn, img, Wid, Hei);
		break;
	default:
		break;
	}
}


//把iplimage转到img中
void IplToImage(IplImage *src)
{
	int w = src->width;
	int h = src->height;
	for(int j = 0; j < h; j++)
	{
		for(int i = 0; i < w; i++)
		{
			img[ (j*w+i)*3 ] = R(src,i,j);
			img[ (j*w+i)*3+1 ] = G(src,i,j);
			img[ (j*w+i)*3+2 ] = B(src,i,j);
		}
	}
}

/*
折半查找，在数组NCumuWeight[N]中寻找一个最小的j，使得
NCumuWeight[j] <=v
float v：              一个给定的随机数
float * NCumuWeight：  权重数组
int N：                数组维数
返回值：
数组下标序号
*/
int binarySearch(float v, float * NCumuWeight, int N)
{
	int low = 0, high = N - 1;
	int n;
	while (low <= high)
	{
		n = (low + high) / 2;
		if (NCumuWeight[n] <= v && NCumuWeight[n+1] > v)
		{
			return n;
		}
		else if (NCumuWeight[n] > v)
		{
			high = n - 1;
		}
		else
		{
			low = n + 1;
		}
	}
	return 0;
}

/*
重新进行重要性采样
输入参数：
float * c：          对应样本权重数组pi(n)
int N：              权重数组、重采样索引数组元素个数
输出参数：
int * ResampleIndex：重采样索引数组
*/
void ImportanceSampling(float * c, int * ResampleIndex, int N)
{
	float rnum, *cumulateWeight;
	cumulateWeight = new float[N+1]; //申请累计权重数组内存

	//cumulateWeight中存放的是weight中的累计权重，最后对累计权重进行归一化
	memset(cumulateWeight, 0.0, sizeof(cumulateWeight));
	for (int i = 0; i < N; i++)
	{
		cumulateWeight[i+1] = cumulateWeight[i] + c[i];
	}
	for (int i = 0; i < N+1; i++)
	{
		cumulateWeight[i] = cumulateWeight[i] / cumulateWeight[N];
	}

	for (int i=0; i<N; i++)
	{
		rnum = (float)rand();
		rnum = rnum / RAND_MAX;
		int j = binarySearch(rnum, cumulateWeight, N+1);
		if(j == N)
		{
			j--;
		}
		ResampleIndex[i] = j;
	}
	delete [] cumulateWeight;
	return;
}

/*
样本选择，从N个输入样本中根据权重重新挑选出N个
输入参数：
SPACESTATE * state：     原始样本集合（共N个）
float * weight：         N个原始样本对应的权重
int N：                  样本个数
输出参数：
SPACESTATE * state：     更新过的样本集
*/
void reSelect(SPACESTATE * state, float * weight, int N)
{
	SPACESTATE *tmpState;
	int i, *rsIdx;
	tmpState = new SPACESTATE[N];
	rsIdx = new int[N];

	ImportanceSampling(weight, rsIdx, N); /* 根据权重重新采样 */

	for (i = 0; i < N; i++)
	{
		tmpState[i] = state[rsIdx[i]];//temState为临时变量,其中state[i]用state[rsIdx[i]]来代替
	}
	for (i = 0; i < N; i++)
	{
		state[i] = tmpState[i];
	}

	delete[] tmpState;
	delete[] rsIdx;

	return ;
}

/*
计算Bhattacharyya系数
输入参数：
float * p, * q：      两个彩色直方图密度估计
int bins：            直方图条数
返回值：
Bhattacharyya系数
*/
float calcuBhattacharyya(float * p, float * q, int bins)
{
	int i;
	float rho;

	rho = 0.0;
	for (i = 0; i < bins; i++)
		rho = (float)(rho + sqrt(p[i]*q[i]));

	return(rho);
}
# define SIGMA2       0.02
float CalcuWeightedPi(float rho)
{
	float pi_n, d2;
	d2 = 1 - rho;
	pi_n = (float)(exp(- d2 / SIGMA2));
	return(pi_n);
}

/*
更新weight数组
输入参数：
SPACESTATE *state：每个粒子的状态
float * weight: 每个粒子所在区域和模板相似度
int N: 粒子数量
unsigned char * image：   图像数据，按从左至右，从上至下的顺序扫描，
颜色排列次序：RGB, RGB, ...				 
int W, H：                图像的宽和高
float * ObjectHist：      目标直方图
int hbins：               目标直方图条数
输出参数：
float * weight：          更新后的权重
返回值：
Bhattacharyya系数
*/
void observe(SPACESTATE * state, float * weight, int N,
			 unsigned char * image, int W, int H,
			 float * ObjectHist, int hbins)
{
	float *ColorHist;
	float rho;
	ColorHist = new float[hbins];

	for (int i=0; i<N; i++)
	{
		// (1) 计算彩色直方图分布
		calcuColorHistogram(state[i].xt, state[i].yt,state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist, hbins);
		// (2) Bhattacharyya系数
		rho = calcuBhattacharyya(ColorHist, ObjectHist, hbins);
		// (3) 根据计算得的Bhattacharyya系数计算各个权重值
		weight[i] = CalcuWeightedPi(rho);
	}
	delete [] ColorHist;
	return ;
}

/*
估计，根据权重，估计一个状态量作为跟踪输出
输入参数：
SPACESTATE * state：      状态量数组
float * weight：          对应权重
int N：                   例子数量
输出参数：
SPACESTATE * EstState：   估计出的状态量
*/
void estimation(SPACESTATE * state, float * weight, int N,
				SPACESTATE & EstState)
{
	int i;
	float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
	float weight_sum;

	at_dot = 0;
	Hxt = 0;
	Hyt = 0;
	v_xt = 0;
	v_yt = 0;
	xt = 0;
	yt = 0;
	weight_sum = 0;
	// 求和
	for (i = 0; i < N; i++)
	{
		at_dot += state[i].at_dot * weight[i];
		Hxt += state[i].Hxt * weight[i];
		Hyt += state[i].Hyt * weight[i];
		v_xt += state[i].v_xt * weight[i];
		v_yt += state[i].v_yt * weight[i];
		xt += state[i].xt * weight[i];
		yt += state[i].yt * weight[i];
		weight_sum += weight[i];
	}

	// 求平均
	if (weight_sum <= 0) weight_sum = 1; // 防止被0除，一般不会发生
	EstState.at_dot = at_dot / weight_sum;
	//EstState.Hxt = (int)(Hxt/weight_sum + 0.5);
	//EstState.Hyt = (int)(Hyt/weight_sum + 0.5);
	EstState.Hxt = (int)(Hxt / weight_sum);
	EstState.Hyt = (int)(Hyt / weight_sum);
	EstState.v_xt = v_xt / weight_sum;
	EstState.v_yt = v_yt / weight_sum;
	//EstState.xt = (int)(xt/weight_sum + 0.5);
	//EstState.yt = (int)(yt/weight_sum + 0.5);
	EstState.xt = (int)(xt / weight_sum);
	EstState.yt = (int)(yt / weight_sum);

	return;
}

/*
更新模型
输入参数：
SPACESTATE EstState:状态量的估计值
float * TargetHist: 模型的颜色直方图
int bins:           直方图条数
float Pit:          模型更新权重
unsigned char *img: 输入图像
int W, H:           图像宽和高
输出：
float *TargetHist:  更新后的模板直方图
*/
void modelUpdate(SPACESTATE EstState, float * TargetHist, int bins, float PiT,
				unsigned char * img, int W, int H)
{
	float *EstHist;	//目标直方图
	float Bha;		//巴氏距离
	float Pi_E;		//置信度
	//预测目标的直方图
	EstHist = new float[bins];
	//在估计值处计算目标直方图
	calcuColorHistogram(EstState.xt, EstState.yt, EstState.Hxt,
		EstState.Hyt, img, W, H, EstHist, bins);
	//计算Bhattacharyya系数
	Bha = calcuBhattacharyya(EstHist, TargetHist, bins);
	//计算概率权重
	Pi_E = CalcuWeightedPi(Bha);
	//如果置信度大于阈值
	if (Pi_E > PiT)
	{
		for (int i=0; i<bins; i++)
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
			+ ALPHA_COEFFICIENT * EstHist[i]);
		}
	}
	delete [] EstHist;
	return;
}

/*
总控程序，完成整个跟踪流程
输入参数：
unsigned char * image:	当前帧图像
int W, int H：			图像的宽和高
int &xc int &yc：		输出结果区域的中心点
int & Wx_h, int & Hy_h：	输出结果区域的半宽和半高
float & maxWeight：		最大相似度
输出：
int &xc int &yc：		输出结果区域的中心点
int & Wx_h, int & Hy_h：	输出结果区域的半宽和半高
float & maxWeight：		最大相似度
*/
int colorParticleTracking(unsigned char * image, int W, int H
						  ,int &xc, int &yc, int & Wx_h, int & Hy_h,
						  float & maxWeight)
{
	//预测的结果
	SPACESTATE EState;
	// 根据权重重采样，相似度达的点会更容易被选到，
	reSelect(states, weights, NParticle);
	// 为重采样的点添加噪声，向周围传播
	propagate(states, NParticle);
	// 观测：更新weight数组
	observe(states, weights, NParticle, image, W, H,
		modelHist, nbin);
	// 估计：对状态量进行估计，提取位置量
	estimation(states, weights, NParticle, EState);
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	//更新模型
	modelUpdate(EState, modelHist, nbin, Pi_Thres, image, W, H);
	// 计算最大相似度
	maxWeight = weights[0];
	for (int i = 1; i < NParticle; i++)
		maxWeight = maxWeight < weights[i] ? weights[i] : maxWeight;

	// 进行合法性检验，不合法返回-1
	if (xc < 0 || yc < 0 || xc >= W || yc >= H ||
		Wx_h <= 0 || Hy_h <= 0)
	{
		return -1 ;
	}
	else
	{
		return 1;
	}
}

int main(int argc, char *argv[])
{
	CvCapture *capture = 0;
	capture = cvCaptureFromAVI("../13.avi");
	//capture = cvCreateCameraCapture(0);
	int row, col;
	float similarity;		//相似度
	float maxWeight;
	bool start = false;

	while (capture)
	{
		curFrame = cvQueryFrame(capture);
		if (curFrame == NULL)
		{
			break;
		}

		//初始化
		if (start == false)
		{
			Hei = curFrame->height;
			Wid = curFrame->width;
			img = new unsigned char [Wid * Hei * 3];
			start = true;
		}
		IplToImage(curFrame);
		maxWeight = 0.0;

		//开始跟踪
		if (track == true)
		{
			similarity = colorParticleTracking(img, Wid, Hei, xOut, yOut, widOut, heiOut, maxWeight);
			if (similarity > 0 && maxWeight > 0.0001)
			{
				cvRectangle(curFrame, cvPoint(xOut - widOut, yOut - heiOut),
					cvPoint(xOut+widOut, yOut+heiOut), cvScalar(255,0,0), 2, 8, 0);
				xIn = xOut; yIn = yOut;
				WidIn = widOut; HeiIn = heiOut;
			}
			else
			{
				cout << "target lost" << endl;
			}
		}

		cvShowImage("vedio", curFrame);
		cvSetMouseCallback("vedio", mouseHandler, 0);
		if (pause)
		{
			cvWaitKey(1000);
		}
		else
		{
			cvWaitKey(10);
		}
	}

	cvReleaseImage(&curFrame);
	clearAll();
	cvDestroyAllWindows();
}