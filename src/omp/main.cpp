#include <iostream>
#include "main.h"
#include <vector>
#include "../third-party/eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;

/* 
 * Eigen学习和Batch OMP的编写
 * Eigen的快速指导手册：http://eigen.tuxfamily.org/dox/group__QuickRefPage.html
 * 或参考：https://blog.csdn.net/wzaltzap/article/details/79501856
 */

VectorXd batch_omp(VectorXd alpha0,double e0,MatrixXd G,double err=1e-6,int k=0){
	/* 所有变量同./omp.py下函数omp_3 */
	int w,n=0;
	double e=e0;
	w=(int)G.rows();
	if(k==0 || k>w){
		k=w;
	}
	MatrixXd L(w,w);
	L.setZero();
	L(0,0)=1;
	VectorXd alpha=alpha0;
	vector<int> I;
	vector<int> I_;
	for(int i=0;i<w;++i)
		I_.push_back(i);
	VectorXd G_Ik_(w);
	VectorXd alpha0_I(w);
	MatrixXd GT_I(w,w);
	VectorXd y(w);
	y.setZero();
	double delta=0;
	for(int n=0;n<k;++n){
		//获取对残差的最大贡献基索引
		int k_=I_[0]; //fix
		double max_val=0;
		int ptr=0; //fix
		for(int i=0;i<I_.size();++i){
			double v=abs(alpha[I_[i]]);
			if(v>max_val){max_val=v;k_=I_[i];ptr=i;}
		}
		I_.erase(I_.begin()+ptr);
		if(n>=1){
			for(int i=0;i<I.size();++i)
				G_Ik_[i]=G(I[i],k_);
			VectorXd w_=L.topLeftCorner(n,n).lu().solve(G_Ik_.head(n));
			L.row(n).head(n)=w_;
			L(n,n)=sqrt(1-w_.squaredNorm());
		}
		
		I.push_back(k_);
		GT_I.row(n)=G.row(k_);
		//解方程组LL'g=alpha0_I
		alpha0_I[n]=alpha0[k_];
		VectorXd Lp_t=L.topLeftCorner(n+1,n+1).lu().solve(alpha0_I.head(n+1));
		VectorXd Lp=L.topLeftCorner(n+1,n+1).transpose().lu().solve(Lp_t);
		VectorXd beta=GT_I.topRows(n+1).transpose()*Lp;
		alpha=alpha0-beta;
		VectorXd beta_I(n+1);
		for(int i=0;i<I.size();++i)
			beta_I[i]=beta[I[i]];
		double t=Lp.transpose()*beta_I;
		e=e-t+delta;
		if(e<err){
			break;
		}
		delta=t;
		for(int i=0;i<I.size();++i)
			y[I[i]]=Lp[i];
	}
	return y;
}

void batch_omp_wrapper(double *_alpha0,double *_G,double *_result,int aL,int Gn,int Gm,double e0,double err=1e-6,int k=0){
	//该函数的作用显而易见，完成C普通数组与Eigen中矩阵、向量对象的转换，即完成与函数batch_omp的对接
	Map<Matrix<double,Dynamic,1>>alpha0(_alpha0,aL,1);
	Map<Matrix<double,Dynamic,Dynamic,RowMajor>>G(_G,Gn,Gm);
	Map<Matrix<double,Dynamic,1>>result(_result,Gm,1);
	result=batch_omp(alpha0,e0,G,err,k);
}

int main(){
	//使用g++编译项目，此处即g++ main.cpp，如果有多个cpp文件的话，需要作为参数全部给出，譬如g++ main.cpp other.cpp，但是多个cpp只应有一个int main()主函数。参考：https://www.cnblogs.com/fenghuan/p/4794514.html
	cout<<"### Batch-OMP ###"<<endl; //C++标准库定义的名字都在命名空间std中，要省略std，则在开头using namespace std，否则std::cout或std::endl
	
	double alpha0[]={-1.3433,1.17,0.25}; //alpha0:=Dᵀx，D、x数值见./omp.py
	int aL=3;
	double G[]={0.999698,-0.1414,-0.707,-0.1414,1,-0.6,-0.707,-0.6,1}; //G:=DᵀD
	int Gn=3;
	int Gm=3;
	double e0=2.785; //e0:=xᵀx
	double result[3]; //存放结果
	
	batch_omp_wrapper(alpha0,G,result,aL,Gn,Gm,e0);
	for(int i=0;i<Gm;i++)cout<<result[i]<<" ";
	cout<<endl;
	return 1;
}

/*
规约和广播：https://blog.csdn.net/u012936940/article/details/79842944
协方差计算：https://www.jianshu.com/p/87bd1766b37b
伪逆计算：https://www.jianshu.com/p/d2a197c930cc
一些类型转换：https://www.cnblogs.com/VVingerfly/p/8037490.html
求解线性方程组：https://www.cnblogs.com/zhuzhudong/p/10827749.html
	https://blog.csdn.net/m0_37579176/article/details/78517185
*/