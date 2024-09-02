#include<stdio.h>

#define M 100

int num = 0;  // 用于记录当前数组中的有效数据个数。

int flag = 1; // 1: true, 0: false

typedef struct Student{
        char name[30];
        int id;
        int age;
        int gender; // 1:男 ，0：女
        char desc[100];
}Stu;

void addStu(Stu *stuP);
void showStu(Stu *stuP);
void updata(Stu *stuP);
void save(Stu *stuP);
void exit();


int main(int argc, char** argv){

        printf("******** 欢迎来到学生信息管理系统！！！********\n");


        Stu stu[M];
        Stu *stuP = &stu[0];


        while(flag){

                printf("\n");
                printf("----------------------------------\n");
                printf("------   1, 增加 学生信息   ------\n");
                printf("------   2, 查看 学生信息   ------\n");
                printf("------   3, 修改 学生信息   ------\n");
                printf("------   4, 保存 学生信息   ------\n");
                printf("------   5, 退出系统        ------\n");
                printf("----------------------------------\n");
                printf("\n");

                int op;

		printf("请输入将要执行的操作 ：");
                scanf("%d",&op);
                printf("\n");printf("\n");


                switch(op){
                        case 1:
                                addStu(stuP);
                                num = num +1;
                                printf("添加成功！\n");
                                break;

                        case 2:
                                showStu(stuP);
                                break;

                        case 3:
                                updata(stuP);
                                break;

                        case 4:
                                save(stuP);
                                printf("saved");
                                break;

                        case 5:
                                exit();
                                break;

                        default: printf("输入有误，请重新输入!\n");

                }



        }

	
	return 0;
}


void addStu(Stu *stuP){

        printf("执行增加学生信息功能：\n");

        printf("请输入学生姓名：");
        scanf("%s",(stuP+num)->name);
        printf("\n");

        printf("请输入学生学号：");
        int idTemp;
        scanf("%d",&idTemp);
        (stuP+num)->id = idTemp;
        printf("\n");

        printf("请输入学生性别（1：男，0：女）：");
        int genderTemp;
        scanf("%d",&genderTemp);
        (stuP+num)->gender = genderTemp;
        printf("\n");

        printf("请输入学生年龄：");
        int ageTemp;
        scanf("%d",&ageTemp);
        (stuP+num)->age = ageTemp;
        printf("\n");

        printf("请输入学生备注：");
        scanf("%s",(stuP+num)->desc);
        printf("\n");

}

void showStu(Stu *stuP){

        printf("学生信息如下：\n");


        for(int i=0;i<num;i++){
                printf("第 %d 名同学的信息如下：\n",i);
                printf("    姓名：%s\n", (stuP+i)->name );
                printf("    学号：%d\n", (stuP+i)->id );
                printf("    性别：%d\n", (stuP+i)->gender );
                printf("    年龄：%d\n", (stuP+i)->age );
                printf("    备注：%s\n", (stuP+i)->desc );
        }


}

void updata(Stu *stuP){

        int index;
        printf("请输入将要修改的学生的序号：");
        scanf("%d",&index);
        if(index > num){
                printf("序号超出范围，程序结束！");
                return;
        }

        printf("请输入学生姓名：");
        scanf("%s",(stuP+index)->name);
        printf("\n");


        printf("请输入学生学号：");
        int idTemp;
        scanf("%d",&idTemp);
        (stuP+index)->id = idTemp;
        printf("\n");

        printf("请输入学生性别（1：男，0：女）：");
        int genderTemp;
        scanf("%d",&genderTemp);
        (stuP+index)->gender = genderTemp;
        printf("\n");

        printf("请输入学生年龄：");
        int ageTemp;
        scanf("%d",&ageTemp);
        (stuP+index)->age = ageTemp;
        printf("\n");

        printf("请输入学生备注：");
        scanf("%s",(stuP+index)->desc);
        printf("\n");

}

void save(Stu *stuP){

        FILE *fpOut;

        //if(( fpOut = fopen("C:/Users/12170/Desktop/1.txt","aw+") ) == NULL){
        if(( fpOut = fopen("./1.txt","aw+") ) == NULL){
                printf("输出文件打开错误。\n");
                return;
        }


        for(int i=0;i<num;i++){
                fprintf(fpOut, "第 %d 名同学的信息如下：\n", i);
                fprintf(fpOut, "    姓名：%s\n", (stuP+i)->name);
                fprintf(fpOut, "    学号：%d\n", (stuP+i)->id);
                fprintf(fpOut, "    性别：%d\n", (stuP+i)->gender);
                fprintf(fpOut, "    年龄：%d\n", (stuP+i)->age);
                fprintf(fpOut, "    备注：%s\n", (stuP+i)->desc);
        }

}

void exit(){
        flag = 0;
}


























