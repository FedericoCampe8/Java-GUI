/************************************************************************
 *               Jnet - A consensus neural network secondary            * 
 *                      structure prediction method                     * 
 *                                                                      *
 *                          James Cuff (c) 1999                         *
 ************************************************************************


                                 LICENCE
-------------------------------------------------------------------------
This software can be copied and used freely providing it is not 
resold in any form and its use is acknowledged.  

This software is provided by "as is" and any express or implied
warranties, including, but not limited to, the implied warranties of
merchantability and fitness for a particular purpose are disclaimed.
In no event shall the regents or contributors be liable for any
direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute
goods or services; loss of use, data, or profits; or business
interruption) however caused and on any theory of liability, whether
in contract, strict liability, or tort (including negligence or
otherwise) arising in any way out of the use of this software, even 
if advised of the possibility of such damage.
-------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "jnet.h"
#include "psinet1.h" 
#include "psinet2.h" 
#include "psinet1b.h"
#include "psinet2b.h"
#include "psinet1c.h" 
#include "psinet2c.h" 
#include "hmm1.h" 
#include "hmm2.h"
#include "net1.h" 
#include "net2.h" 
#include "net1b.h"
#include "net2b.h"
#include "consnet.h"
#include "hmmsol25.h" 
#include "psisol25.h"
#include "psisol0.h" 
#include "hmmsol0.h"
#include "psisol5.h"
#include "hmmsol5.h"

FILE *fsec;
FILE *psifile;
FILE *hmmfile;
FILE *psifile2;

typedef struct alldata {
  int   **seqs;
  int   *secs;
  float conserv[MAXSEQLEN];
  float cav;
  float constant;
  float smcons[MAXSEQLEN];
  int   numseq;
  int   segdef[4][MAXSEQLEN]; 
  int   segbin[4][100]; 
  int   numsegs[4]; 
  int   profile[MAXSEQLEN][24];
  int   posn[MAXSEQLEN]; 
  int   lens;
  int   profmat[MAXSEQLEN][24];
  int   psimat2[MAXSEQLEN][20]; 
  float psimat[MAXSEQLEN][20]; 
  float hmmmat[MAXSEQLEN][24]; 
}alldata;

void  seq2int(char letseq[MAXSEQLEN], int seq[MAXSEQLEN], int length);
void  int2sec(int sec[MAXSEQLEN], char letsec[MAXSEQLEN], int length);
void  int2seq(int seq[MAXSEQLEN], char letseq[MAXSEQLEN], int length);
void  int2pred(int pred[MAXSEQLEN], char letpred[MAXSEQLEN], int length);
void  pred2int(int pred[MAXSEQLEN], char letpred[MAXSEQLEN], int length);
void  getone(FILE *fsec, int seq[MAXSEQLEN], int *lenseq);
void  getsec(FILE *fsec, int sec[MAXSEQLEN], int *lensec);
float doconf (float confa, float confb, float confc);

void  dowinss (
	       float seq2str[MAXSEQLEN][3], 
	       int arlen, 
	       int win, 
	       int curpos, 
	       float winarss[30][4], 
	       int m, 
	       alldata *data[]
	       );

void dowinsspsi (
		 float seq2str[MAXSEQLEN][3], 
		 int arlen, 
		 int win, 
		 int curpos, 
		 float winarss[30][4], 
		 int m, 
		 alldata *data[]
		 );

void  doprofwin (
		 alldata *data[], 
		 int arlen, 
		 int win, 
		 int curpos, 
		 int winar2[30][30], 
		 int m,
		 int whichmat
		 );

void doprofpsi2 (
		alldata *data[], 
		int arlen, 
		int win, 
		int curpos, 
		int winar2[30][30],
		int m, 
		int whichmat);

void doprofpsi (
		alldata *data[], 
		int arlen, 
		int win, 
		int curpos, 
		float psiar[30][30],
		int m, 
		int whichmat
		);
void doprofhmm(
	       alldata *data[], 
	       int arlen, 
	       int win, 
	       int curpos, 
	       float psiar[30][30],
	       int m, 
	       int whichmat
	       );
void filter (char inp[MAXSEQLEN], char outp[MAXSEQLEN]);
void printstring (
		  char *seqdef, 
		  char pred[MAXSEQLEN], 
		  char jury[MAXSEQLEN],
		  char alignlet[MAXSEQLEN],
		  char hmmlet[MAXSEQLEN],
		  char psi1let[MAXSEQLEN],
		  char psi2let[MAXSEQLEN],
		  float confidence[MAXSEQLEN], 
		  int len,
		  int predmode,
		  char sollet25[MAXSEQLEN],
	          char sollet5[MAXSEQLEN],
		  char sollet0[MAXSEQLEN]
		  );

void doprofile(
	       alldata *data[], 
	       int count
	       ); 
void doposn(
	     alldata *data[], 
	     int count
	     ); 
void doprofilemat (
		    alldata *data[], 
		    int count, 
		    int matrix[24][24]
		    ); 
void dolens(
	     alldata *data[], 
	     int count
	     ); 
void StrReplStr(
		char *s1,
		char *s2,
		char *FromSStr, 
		char *ToSStr
		);

char  *Strncat(char *s1,char *s2,int len);
void  readpsi(FILE *psifile, alldata *data[]);
void  readpsi2(FILE *psifile2, alldata *data[]);
void  readhmm(FILE *hmmfile, alldata *data[]);
void  defcons (alldata *data[], int count);
int   countseqs(FILE *fsec);
void  dopred(alldata *data[], int count, int printsty);
void  outputsequenceprofile(alldata *data[], int count);
void  outputstructureprofile(alldata *data[], int count);
void  outputstructureprofilepsi(alldata *data[], int count);


int main(argc, argv)
     int     argc;
     char    *argv[];
    
{
  
  float aveaccq3,aveacclen,aveacccomb;
  int seq[MAXSEQLEN];
  int sec[MAXSEQLEN];
  int numseqs;
  int current;
  int opt;
  int n,i,j,y,m,k,x,z,q,junk,count,allcount;
  int length;
  float aveacc,bestacc,obestacc,acc1,acc2;
  int startchange,stopchange;
  int xwinbesta,xwinbestb,xwinbestc; 
  int succ,totsucc,notsucc,converge;
  int report;
  int printer,mult; 
  int whichone,loopcount,writecount; 
  int same;
  float lastacc;
  int lensec,lenseq;
  int mat,justpred;
  int dcsave[3]; 
  char *st;
  char *fp;
  int predmode;
 
  alldata *data [MAXSEQNUM];
  count=0;
  loopcount=0;writecount=0;
  printer=0;x=0;
  count=1;
  lensec=lenseq=0;
  junk=0;
  justpred=predmode=0;
  obestacc=0;

  nopsi=0;
  nohmm=0;
  aveaccq3=aveacclen=aveacccomb=0;

 
  if (argc < 3) {
    fprintf(stdout, "\n\nJnet - secondary structure prediction method\n\nUsage: jnet -mode <sequence file> [hmm profile] [<psiblast pssm> and <psiblast freq>]\n\n\tHMM profile and PSIBLAST profiles are optional\n\tPSIBLAST profiles must be supplied in pairs\n\nModes:\t-p Human readable\n\t-c Concise output\n\t-z Column output\n\n");
    exit(-1);
  }
  if ((fsec = fopen (argv[2],"r")) == NULL){
    //fprintf (stderr, "ERROR: Can't open alignment file %s\nexit()\n",argv[2]);
    exit(0);
  }
  else {fsec = fopen (argv[2],"r");}
  
  if ((psifile = fopen (argv[3],"r")) == NULL){
    //fprintf (stderr, "Warning! : Can't open HMM profile file\nFalling back to less accurate alignment mode\n");
    nohmm=1;
  }
  if (nohmm == 0){
    hmmfile = fopen (argv[3],"r");
  } 
    
  if ((psifile = fopen (argv[4],"r")) == NULL){
    //fprintf (stderr, "Warning! : Can't open PSIBlast profile file\nFalling back to less accurate alignment mode\n");
    nopsi=1;
  }

  if (nopsi == 0){
    psifile = fopen (argv[4],"r");
    
    if ((psifile2 = fopen (argv[5],"r")) == NULL){
       //fprintf (stderr, "ERROR!\nCan't open second psiblast profile - need both to work\nRun getfreq *and* getpssm on the psiblast report file\n\n");
       exit(0);
    }
    psifile2 = fopen (argv[5],"r");
  }
    
   st=argv[1] + 1;
	
   if (*st == 'p'){/*fprintf (stderr, "\nMODE: Prediction\n");*/ opt=1;}
   if (*st == 'z'){/*fprintf (stderr, "\nMODE: Prediction - Computer output\n");*/ opt=4;}
   if (*st == 'c'){/*fprintf (stderr, "\nMODE: Prediction - Concise output\n");*/ opt=5;} 
   if (*st == 'h'){/*fprintf (stderr, "\nMODE: Prediction - HTML output\n");*/ opt=6;}
	 
   if (opt == 0)  {
     fprintf (stderr,"ERROR: Invalid option selected\nERROR: Please select an option\n\n\n");
     fprintf (stderr,"-p -> prediction mode \n");
     fprintf (stderr,"-z -> prediction mode - computer output\n"); 
     exit(0);
   } 


   //fprintf(stderr, "JNet Started!\nReading Data\n");
   data[0]=(alldata *) malloc (sizeof (alldata));
   numseqs = countseqs(fsec);
   data[0]->seqs = (int **) malloc(sizeof(int *)* numseqs);
   //fprintf (stderr, "There are %d sequence homologues in the file\n",numseqs);
   fclose (fsec);
   fsec = fopen (argv[2],"r");
   for (j = 0;j < numseqs ;j++){ 
     lenseq=0;
     getone(fsec,seq,&lenseq);
     data[0]->seqs[j] = (int *) malloc(sizeof(int)*(lenseq+1));   
     for (i = 0;i <= lenseq ;i++){     
       data[0]->seqs[j][i]=seq[i];
     }       
     data[0]->lens=lenseq; 
   } 
   
   
   data[0]->numseq=numseqs;
   //fprintf (stderr, "\nGenerating...\n");
   //fprintf (stderr, "\tLength numbers\n"); 
   dolens(data,count);
   //fprintf (stderr, "\tProfile - frequency based\n");
   doprofile(data,count);
   //fprintf (stderr, "\tProfile - average mutation score based\n"); 
   doprofilemat(data,count,matrix);
   //fprintf (stderr, "\tConservation numbers\n"); 
   defcons(data,count); 
   fprintf (stderr, "Done initial calculations!\n\n"); 
   
   
   if (nohmm == 0){
     fprintf (stderr, "Found HMM profile file...\nUsing HMM enhanced neural networks\n\n");
     readhmm(hmmfile, data);
   }
   if (nopsi == 0){
     fprintf (stderr, "Found PSIBlast profile files...\nUsing PSIBlast enhanced neural networks\n\n");
     readpsi(psifile, data);
     readpsi2(psifile2,data);
   }   
   if (opt == 1){
     fprintf (stderr,"Running final predictions!\n");
     dopred(data,count,0);
     fprintf (stderr,"All done!\n");
     exit(0);
   }
   if (opt == 4){
     fprintf (stderr,"Running final predictions!\n");
     dopred(data,count,1);
     fprintf (stderr,"All done!\n");
     exit(0);
   }
   
   if (opt == 5){
     fprintf (stderr,"Running final predictions!\n");
     dopred(data,count,2);                          
     fprintf (stderr,"All done!\n");                
     exit(0);                      
   }   
   if (opt == 6){
     fprintf (stderr,"Running final predictions!\n");
     dopred(data,count,3); 
     fprintf (stderr,"All done!\n");                
     exit(0);                                      
   }      
}



int countseqs(FILE *fsec)
{
  int i;
  int c;
  
  i=c=0;
  
  while((c=getc(fsec)) != EOF){
    if (c == '>'){
      i++;
    }
  } 
  return i;
}


void defcons(alldata *data[],int count)
     
{
  int m,i,x,j,p,z,len;
  int tot[10]={0};
  int cons1[MAXSEQNUM]={0};
  int constab[MAXSEQNUM][24]={0};
  float ci,cav,aveci,avci;
  p=0;
  ci=0.0;cav=0.0;aveci=0.0;
  len=0;
  for (m=0;m < count;m++){  
    for (i=0;data[m]->seqs[0][i] != 25;i++){  
      
      for(x=0;x < (data[m]->numseq) ;x++){        
	cons1[x] = data[m]->seqs[x][i];      
	for (j=0;j<=9;j++){
	  constab[x][j] = ventab[cons1[x]][j]; 
	  tot[j] = tot[j] + constab[x][j];
	}
      }
      for (j=0;j<=9;j++){
	if (tot[j] == data[m]->numseq || tot[j] ==0 ){
	  p++;
	}
	tot[j] = 0; 
      }
     ci = (0.1) * p;
     aveci=aveci+ci;
     p=0;  
     data[m]->conserv[i] = ci;
     len++;
    }
    cav=aveci/len;
    data[m]->cav = cav; 
    if  (data[m]->cav  <= 0.55){
      data[m]->constant = 150;
    }
    if  (data[m]->cav  > 0.55){
      data[m]->constant = 250;
    }   
    ci=0.0;cav=0.0;aveci=0.0;
  }
}

void doprofile (alldata *data[], int count)
     
{
  
  int rescount[24]={0};
  int j,m,i,k;
  int length;
  float floater;
  
  for (m=0;m < count;m++){  
    length=0;
    for (i=0; data[m]->seqs[0][i] != 25; i++){ 
      length++;
    } 
    for (i=0;i < length ; i++){ 
      for (k=0 ; k < 22 ; k++){
	rescount[k]=0;
      }           
      for (j=0 ; j < (data[m]->numseq) ; j++){ 
	
	rescount[data[m]->seqs[j][i]]++;	
      }
      for (k=0 ; k < 22 ; k++){
	floater=rescount[k];
	data[m]->profile[i][k]=((floater/(1.0*data[m]->numseq))*100)+0.5;
      }	      
    }    
  }
}

void dolens (alldata *data[], int count)
     
{
  
  int j,m,i,k;
  int thispos,halfway;
  int length;
  float floater;
  
  for (m=0;m < count;m++){  
    length=0;
    for (i=0; data[m]->seqs[0][i] != 25; i++){ 
      length++;
    } 
    data[m]->lens = length;
  }
}

  
void doprofilemat (alldata *data[], int count, int matrix[24][24])

{

  int j,m,i,k,x;
  int length;
  float floater;
  int tmp;
  
  for (m=0;m < count;m++){
    for (i=0;i < data[m]->lens; i++){
      for (k=0 ; k < 24; k++){        
	data[m]->profmat[i][k] = 0;
      }
    }
  }
  for (m=0;m < count;m++){  
    for (i=0;i < data[m]->lens; i++){
      for (k=0 ; k < 24; k++){     
        for (j=0 ; j < (data[m]->numseq) ; j++){  
	  data[m]->profmat[i][k] = data[m]->profmat[i][k] + matrix[data[m]->seqs[j][i]][k];          
        }       
        data[m]->profmat[i][k] =  data[m]->profmat[i][k]/data[m]->numseq;
      }  
    }
  } 
}




void dowinss (float seq2str[MAXSEQLEN][3], int arlen, int win, int curpos, float winarss[30][4], int m, alldata *data[])
{
  int i,j,k;
  float addon;
  
  j=k=i=0;
  
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){   
    for (k=0;k < 3;k++){
      winarss[j][k]=seq2str[i][k];
    }
    if (i >= 0 && i <= arlen ){        
      
      winarss[j][3]=data[m]->conserv[i];
    }  
    
    if (i < 0 ){    
      winarss[j][0]=0.0000;
      winarss[j][1]=0.0000;
      winarss[j][2]=1.0000;
      winarss[j][3]=0.0000;    
    }   
    if (i >= arlen ){  
      winarss[j][0]=0.0000;
      winarss[j][1]=0.0000;
      winarss[j][2]=1.0000;
      winarss[j][3]=0.0000;
    }       
    j++;
  }
}


void dowinsspsi (float seq2str[MAXSEQLEN][3], int arlen, int win, int curpos, float winarss[30][4], int m, alldata *data[])
{
  int i,j,k;
  float addon;

  j=k=i=0;
  
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){   
    for (k=0;k < 3;k++){
      winarss[j][k]=seq2str[i][k];
    }
  
    if (i < 0 ){    
      winarss[j][0]=0.0000;
      winarss[j][1]=0.0000;
      winarss[j][2]=1.0000;
      
    }   
    if (i >= arlen ){  
      winarss[j][0]=0.0000;
      winarss[j][1]=0.0000;
      winarss[j][2]=1.0000;
    
    }       
    j++;
  }
}
void doposn (alldata *data[], int count)

{
 
  int j,m,i,k; 
  int thispos,halfway; 
  int length; 
  float floater; 
  
  for (m=0;m < count;m++){ 
    length=0; 
    for (i=0; data[m]->seqs[0][i] != 25; i++){  
      length++; 
    }  
    halfway=(length)/2;
    for (i=1;i <= length ; i++){
      if (i <= halfway){
	thispos=i-1;
      }     
      if (i > halfway){
	thispos=(halfway-(i-halfway));
      }
      data[m]->posn[i-1]=thispos;
    }
  } 
}




void dopred(alldata *data[],int count,int printsty)
     
{    
  float seq2str[MAXSEQLEN][3]={0};
  int str2str[MAXSEQLEN]={0};
  int seq2sol[MAXSEQLEN]={0};  
  int predfinalpsi[MAXSEQLEN];
  int filtered[MAXSEQLEN]={0};
  int predarray[7]={0};
  int predfinal[MAXSEQLEN];
  float alignnet[MAXSEQLEN][3]={0};
  float psi1net[MAXSEQLEN][3]={0};
  float psi2net[MAXSEQLEN][3]={0};
  float hmmnet[MAXSEQLEN][3]={0};
  
  float finalout[3][MAXSEQLEN]={0};
  int alignfin[MAXSEQLEN];
  int psi1fin[MAXSEQLEN];
  int psi2fin[MAXSEQLEN];
  int hmmfin[MAXSEQLEN];
  int consfin[MAXSEQLEN];

  char alignlet[MAXSEQLEN];
  char psi1let[MAXSEQLEN];
  char psi2let[MAXSEQLEN];
  char hmmlet[MAXSEQLEN];
  char conslet[MAXSEQLEN];
  char finlet[MAXSEQLEN];


  char sollet25[MAXSEQLEN]; 
  char sollet5[MAXSEQLEN]; 
  char sollet0[MAXSEQLEN]; 

  float conswin[400]={0};
  float consout[3]={0};

  float keepalign[MAXSEQLEN][3]={0}; 
  float netprofin3[500]; 
  float confidence[MAXSEQLEN]={0}; 
  float maxout,maxoutc, maxnext; 
  int profar[23]={0}; 
  float psiar[30][30]; 
  int r;
  int winar2[30][30];

  float solacc25[MAXSEQLEN][2];
  float solacc5[MAXSEQLEN][2]; 
  float solacc0[MAXSEQLEN][2];

  float winarss[30][4]={0}; 
  float netin[400]={0};
  float netprofin[500]={0.0}; 
  float netout[3]={0};
  float netout2[3]={0};
  float neti[400]={0}; 
  char letsec[MAXSEQLEN], letseq[MAXSEQLEN],letpred[MAXSEQLEN],letfilt[MAXSEQLEN];
  int preds[MAXSEQNUM][MAXSEQLEN]; 
  int i,y,j,m,k,z,l,x;
  int pos1,pos2,pos3,pos4; 
  float acc,runacc,aveacc; 
  int cons[MAXSEQNUM];
  int predcount;
  int length;
  int stra,strb,strc;
  int len,cur,residue,seqpos,t;
  float max;
  float ga,gb,gc;
  float addon;
  float confa;
  float confb; 
  float confc; 
  char jury[MAXSEQLEN];
  float outconf;
  int windows;
  int acount,bcount,ccount;
  runacc=0;cur=0;
  letfilt[0] = '\0';
  t=r=0;
  x=y=0;
  m=0;

  
  length=0; 
  for (i=0; data[m]->seqs[0][i] != 25; i++){ 
    length++;
  }
  for (t=0; t < 2; t++){  
    windows=17;   
    for (i=0;i < length;i++){        
      if (t != 1){
	doprofwin(data,length,windows,i,winar2,m,1);
      }
      if (t == 1){
	doprofwin(data,length,windows,i,winar2,m,0);
      }
      j=0;
      for (y=0; y < windows; y++){  
	for (l=0; l < 25; l++){
	  
	  netprofin[j]=winar2[y][l];
	  j++;	  
	}	 
      }
      if (t == 0){net1(netprofin,netout,0);}
      if (t == 1){net1b(netprofin,netout,0);}
      
      if (i <= 4){
	netout[2] = netout[2] + (5-i)*0.2;
      } if (i >= (length-5)){ 
	netout[2] = netout[2] + (5-((length-1)-i))*0.2;	   
      }     
      
      
      seq2str[i][0] = netout[0];
      seq2str[i][1] = netout[1];
      seq2str[i][2] = netout[2];
    }
    
    windows=19;
    
    for (i=0; i < length;i++){      
      dowinss(seq2str,length,windows,i,winarss,m,data);  
      j=0;
      for (y=0; y < windows; y++){
	for (z=0; z < 4; z++){
	  
	  netin[j]=winarss[y][z];
	  j++;
	}	
      }           
      if (t == 0){net2(netin, netout,0);}
      if (t == 1){net2b(netin, netout,0);}
      alignnet[i][0]=(alignnet[i][0]+netout[1]);
      alignnet[i][1]=(alignnet[i][1]+netout[0]);
      alignnet[i][2]=(alignnet[i][2]+netout[2]);       
    }
  }
  for (i=0; i < length;i++){   
    alignnet[i][0]=(alignnet[i][0]/2);
    alignnet[i][1]=(alignnet[i][1]/2);
    alignnet[i][2]=(alignnet[i][2]/2);
  }
  
  
  
  
  for (i=0;i < length;i++){ 
    windows=17;
    doprofhmm(data,length,windows,i,psiar,m,1);
    j=0;
    
    for (y=0; y < windows; y++){  
      for (l=0; l < 24; l++){	  	 
	netprofin3[j]=psiar[y][l];
	j++;	  
      }	 
    }
    hmm1(netprofin3,netout2,0); 
    
    seq2str[i][0] = netout2[0];
    seq2str[i][1] = netout2[1];
    seq2str[i][2] = netout2[2];
  }
  windows=19;
  
  for (i=0; i < length;i++){      
    dowinsspsi(seq2str,length,windows,i,winarss,m,data);  
    j=0;
    for (y=0; y < windows; y++){
      for (z=0; z < 3; z++){    
	netin[j]=winarss[y][z];
	j++;
      }	
    }
    
    hmm2(netin, netout, 0);    
    hmmnet[i][0]=netout[0];
    hmmnet[i][1]=netout[1];
    hmmnet[i][2]=netout[2];
    
  }
  
  if (nohmm ==0){ 
    for (i=0;i < length;i++){
      windows=17;
      doprofhmm(data,length,windows,i,psiar,m,1);
      j=0;
      
      for (y=0; y < windows; y++){
        for (l=0; l < 24; l++){
          netprofin3[j]=psiar[y][l];
          j++;
        }     
      } 
      
      hmmsol25(netprofin3,netout2,0);
      solacc25[i][0] = netout2[0];
      solacc25[i][1] = netout2[1];
      
      hmmsol5(netprofin3,netout2,0);
      solacc5[i][0] = netout2[0];
      solacc5[i][1] = netout2[1];
      
      hmmsol0(netprofin3,netout2,0);
      solacc0[i][0] = netout2[0];
      solacc0[i][1] = netout2[1];
    }
  }
  
  if (nopsi == 0){

    for (i=0;i < length;i++){
      windows=17;
      doprofpsi(data,length,windows,i,psiar,m,1);
      j=0;
      
      for (y=0; y < windows; y++){          
        for (l=0; l < 20; l++){                                  
          netprofin3[j]=psiar[y][l];                                  
          j++;                                    
        }                                         
      }                
      
      psisol25(netprofin3,netout2,0);              
      solacc25[i][0] = solacc25[i][0] + netout2[0];                                 
      solacc25[i][1] = solacc25[i][1] + netout2[1];                                 
      
      psisol5(netprofin3,netout2,0);
      solacc5[i][0] = solacc5[i][0]+ netout2[0];
      solacc5[i][1] = solacc5[i][1] + netout2[1];   
      
      psisol0(netprofin3,netout2,0);
      solacc0[i][0] = solacc0[i][0] + netout2[0];
      solacc0[i][1] = solacc0[i][1] + netout2[1];  
    }                                 
  }
  

  
  for (i=0;i < length;i++){
    if (solacc25[i][0] > solacc25[i][1]){
      sollet25[i]='-';
    }
    if (solacc25[i][1] > solacc25[i][0]){
      sollet25[i]='B';
    } 
    if (solacc5[i][0] > solacc5[i][1]){
      sollet5[i]='-';
    }       
    if (solacc5[i][1] > solacc5[i][0]){
      sollet5[i]='B';
    }    
    if (solacc0[i][0] > solacc0[i][1]){
      sollet0[i]='-';
    }       
    if (solacc0[i][1] > solacc0[i][0]){
      sollet0[i]='B';
    }         
  }
 
  if (nopsi == 0){
    
    for (i=0;i < length;i++){ 
      windows=17;
      doprofpsi2(data,length,windows,i,winar2,m,1);
      j=0;
      
      for (y=0; y < windows; y++){  
	for (l=0; l < 20; l++){	  	 
	  netprofin[j]=winar2[y][l];
	  j++;	  
	}	 
      }
      psinet1(netprofin,netout2,0);  
      seq2str[i][0] = netout2[0];
      seq2str[i][1] = netout2[1];
      seq2str[i][2] = netout2[2];
    }
    windows=19;
    
    for (i=0; i < length;i++){      
      dowinsspsi(seq2str,length,windows,i,winarss,m,data);  
      j=0;
      for (y=0; y < windows; y++){
	for (z=0; z < 3; z++){    
	  netin[j]=winarss[y][z];
	  j++;
	}	
      }
      psinet2(netin, netout, 0);    
      psi1net[i][0]=netout[0];
      psi1net[i][1]=netout[1];
      psi1net[i][2]=netout[2];
    }
    
    
    for (t=0; t < 2; t++){ 
      for (i=0;i < length;i++){
	windows=17;
	doprofpsi(data,length,windows,i,psiar,m,1);
	j=0;	
	for (y=0; y < windows; y++){  
	  for (l=0; l < 20; l++){	  	 
	    netprofin[j]=psiar[y][l];
	    j++;	  
	  }	 
	}	
	if (t == 0){
	  psinet1b(netprofin,netout2,0);
	}
	if (t == 1){
	  psinet1c(netprofin,netout2,0);
	}
		

	seq2str[i][0] = netout2[0];
	seq2str[i][1] = netout2[1];
	seq2str[i][2] = netout2[2];      
      }
      windows=19;
      
      for (i=0; i < length;i++){      
	dowinsspsi(seq2str,length,windows,i,winarss,m,data);  
	j=0;
	for (y=0; y < windows; y++){
	  for (z=0; z < 3; z++){    
	    netprofin[j]=winarss[y][z];
	    j++;
	  }	
	}
	
	if (t == 0){
	  psinet2b(netprofin,netout,0);
	}
	if (t == 1){
	  psinet2c(netprofin,netout,0);
	}
      	
	psi2net[i][0]=(psi2net[i][0]+netout[0]);
	psi2net[i][1]=(psi2net[i][1]+netout[1]);
	psi2net[i][2]=(psi2net[i][2]+netout[2]);
      }
    }
    
    for (i=0; i < length;i++){ 
      psi2net[i][0]=(psi2net[i][0]/2);
      psi2net[i][1]=(psi2net[i][1]/2);
      psi2net[i][2]=(psi2net[i][2]/2);
    }
  }
  
  
  
  for (i=0; i < length;i++){
	
    finalout[0][i]=alignnet[i][0];
    finalout[1][i]=alignnet[i][1];
    finalout[2][i]=alignnet[i][2];
    
    if (i <= 4){
      finalout[2][i] = finalout[2][i] + (5-i)*0.2;
      
    } if (i >= (length-5)){ 
      finalout[2][i] = finalout[2][i] + (5-((length-1)-i))*0.2;    
      
    }     
    
    if (finalout[0][i] > finalout[1][i] && finalout[0][i] > finalout[2][i]  ){
      maxoutc=0;
      maxout=finalout[0][i];
      alignfin[i]=2;
    }
    
    if (finalout[1][i] > finalout[0][i] && finalout[1][i] > finalout[2][i]  ){
      maxoutc=1;
      maxout=finalout[1][i];
      alignfin[i]=1;
    }
    
    if (finalout[2][i] > finalout[0][i] && finalout[2][i] > finalout[1][i]  ){
      maxoutc=2;
      maxout=finalout[2][i];
      alignfin[i]=3;
    }      
  }
  for (i=0; i < length;i++){
    
    finalout[0][i]=psi1net[i][0];
    finalout[1][i]=psi1net[i][1];
    finalout[2][i]=psi1net[i][2];
    
    if (i <= 4){
      finalout[2][i] = finalout[2][i] + (5-i)*0.2;
      
    } if (i >= (length-5)){ 
      finalout[2][i] = finalout[2][i] + (5-((length-1)-i))*0.2;    
      
    }     
    
    if (finalout[0][i] > finalout[1][i] && finalout[0][i] > finalout[2][i]  ){
      maxoutc=0;
      maxout=finalout[0][i];
      psi1fin[i]=2;
    }
    
    if (finalout[1][i] > finalout[0][i] && finalout[1][i] > finalout[2][i]  ){
      maxoutc=1;
      maxout=finalout[1][i];
      psi1fin[i]=1;
    }
    
    if (finalout[2][i] > finalout[0][i] && finalout[2][i] > finalout[1][i]  ){
      maxoutc=2;
      maxout=finalout[2][i];
      psi1fin[i]=3;
    }      
  }
  for (i=0; i < length;i++){
    
    finalout[0][i]=psi2net[i][0];
    finalout[1][i]=psi2net[i][1];
    finalout[2][i]=psi2net[i][2];
    
    if (i <= 4){
      finalout[2][i] = finalout[2][i] + (5-i)*0.2;
      
    } if (i >= (length-5)){ 
      finalout[2][i] = finalout[2][i] + (5-((length-1)-i))*0.2;    
      
    }     
    
    if (finalout[0][i] > finalout[1][i] && finalout[0][i] > finalout[2][i]  ){
      maxoutc=0;
      maxout=finalout[0][i];
      psi2fin[i]=2;
    }
    
    if (finalout[1][i] > finalout[0][i] && finalout[1][i] > finalout[2][i]  ){
      maxoutc=1;
      maxout=finalout[1][i];
      psi2fin[i]=1;
    }
    
    if (finalout[2][i] > finalout[0][i] && finalout[2][i] > finalout[1][i]  ){
      maxoutc=2;
      maxout=finalout[2][i];
      psi2fin[i]=3;
    }      
  }
  for (i=0; i < length;i++){
    
    finalout[0][i]=hmmnet[i][0];
    finalout[1][i]=hmmnet[i][1];
    finalout[2][i]=hmmnet[i][2];
    
    if (i <= 4){
      finalout[2][i] = finalout[2][i] + (5-i)*0.2;
      
    } if (i >= (length-5)){ 
      finalout[2][i] = finalout[2][i] + (5-((length-1)-i))*0.2;    
      
    }     

    if (finalout[0][i] > finalout[1][i] && finalout[0][i] > finalout[2][i]  ){
      maxoutc=0;
      maxout=finalout[0][i];
      hmmfin[i]=2;
    }
    
    if (finalout[1][i] > finalout[0][i] && finalout[1][i] > finalout[2][i]  ){
      maxoutc=1;
      maxout=finalout[1][i];
      hmmfin[i]=1;
    }
    
    if (finalout[2][i] > finalout[0][i] && finalout[2][i] > finalout[1][i]  ){
      maxoutc=2;
      maxout=finalout[2][i];
      hmmfin[i]=3;
    }      
  }
       
  int2pred(alignfin,alignlet,length);
  int2pred(psi1fin,psi1let,length);
  int2pred(psi2fin,psi2let,length);
  int2pred(hmmfin,hmmlet,length);
  
  if (nohmm ==0 && nopsi==0){
    for (j=0; j < length;j++){
      if (alignlet[j] != psi1let[j] || alignlet[j] != psi2let[j] || alignlet[j] != hmmlet[j]){
	r=0;
	for (y=-8; y <= 8; y++){
	  x=j+y;
	  if (x > 0 && x < length){
	    
	    conswin[r]= alignnet[x][0];r++;
	    conswin[r]= alignnet[x][1];r++;
	    conswin[r]= alignnet[x][2];r++;
	    conswin[r]= psi1net[x][0];r++;
	    conswin[r]= psi1net[x][1];r++;
	    conswin[r]= psi1net[x][2];r++;
	    conswin[r]= psi2net[x][0];r++;
	    conswin[r]= psi2net[x][1];r++;
	    conswin[r]= psi2net[x][2];r++;
	    conswin[r]= hmmnet[x][0];r++;
	    conswin[r]= hmmnet[x][1];r++;
	    conswin[r]= hmmnet[x][2];r++;
	    
	  }
	  else {
	    for (z=0;z<12;z++){
	      conswin[r]=0;r++;
	    } 
	  }
	  
	}
	
	consnet(conswin, consout, 0);
        
	finalout[0][j]=consout[0];
	finalout[1][j]=consout[1];
	finalout[2][j]=consout[2];
      }
      
     consfin[j]=3;
     
     if (finalout[0][j] > finalout[1][j] && finalout[0][j] > finalout[2][j]  ){
       consfin[j]=1;
     }
     
     if (finalout[1][j] > finalout[0][j] && finalout[1][j] > finalout[2][j]  ){
       consfin[j]=2;
     }
     
     if (finalout[2][j] > finalout[0][j] && finalout[2][j] > finalout[1][j]  ){
       consfin[j]=3;
     }
     
     
     confidence[j] = doconf(consout[0],consout[1],consout[2]);   
     jury[j] = '*';
     
     if (alignlet[j] == psi1let[j] && alignlet[j] == psi2let[j] && alignlet[j] == hmmlet[j]){
       jury[j] = ' ';
       consfin[j]=psi2fin[j];
       
       confidence[j] = doconf(psi2net[j][0],
			      psi2net[j][1],
			      psi2net[j][2]); 
     }
    }
    consfin[length]=25;
  }
  
  
  
  
  if (nohmm == 1 && nopsi == 1){
    fprintf (stderr, "\nWARNING!: Only using the sequence alignment\n          Accuracy will average 71.6%\n\n");
    for (j=0; j < length;j++){
      consfin[j]=alignfin[j]; 
      
      confidence[j] = doconf((alignnet[j][0]),(alignnet[j][1]),(alignnet[j][2]));
    }
    consfin[length]=25;
    
  }
  
  
  if (nohmm == 0 && nopsi ==1){
    fprintf (stderr, "\n\nWARNING!: Only using the sequence alignment, and HMM profile\n          Accuracy will average 74.4%\n\n");
    for (j=0; j < length;j++){
      consfin[j]=hmmfin[j];
      
      confidence[j] = doconf((hmmnet[j][0]),(hmmnet[j][1]),(hmmnet[j][2]));
    }
    consfin[length]=25;
  }
  
  
  if (nohmm == 0 && nopsi == 0){
  fprintf (stderr, "\n\nBoth PSIBLAST and HMM profiles were found\nAccuracy will average 76.4%\n\n");
  }
  
  
  int2pred(consfin,finlet,length);
  filter(finlet,letfilt);
  filter(finlet,letfilt);
  filter(finlet,letfilt);
  int2seq(data[m]->seqs[0],letseq,length);
  len=65;
  
  
  
  
  
  
  if (nohmm == 1 && nopsi ==1){
    if (printsty==0){
      printf("\nLength = %2d  Homologues = %2d\n",length,data[m]->numseq);
      printstring(letseq,letfilt,jury,alignlet,hmmlet,psi1let,psi2let,confidence,len,0,sollet25,sollet5,sollet0);
    }
    
    
    if (printsty==1){
    printf ("START PRED\n");
    for (j=0; j < length;j++){
      printf ("%c %c | %c %c %1.0f %5.5f %5.5f %5.5f\n",letseq[j],letfilt[j],alignlet[j],jury[j],confidence[j],alignnet[0][j],alignnet[1][j],alignnet[2][j]);
    }
    printf ("END PRED\n");
    }
    
    
    if (printsty==2){
      printf ("\njnetpred:");
      for (j=0; j < length;j++){
      printf ("%c,",letfilt[j]);
      } 
      
      printf ("\nJNETPROPH:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",alignnet[1][j]);
      } 
      printf ("\nJNETPROPB:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",alignnet[0][j]);
      } 
      printf ("\nJNETPROPC:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",alignnet[2][j]);
    }
      printf ("\nJNETCONF:");
      for (j=0; j < length;j++){
	printf ("%1.0f,",confidence[j]);
      }
      printf ("\nJNETSOL25:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet25[j]);
      }
      printf ("\nJNETSOL5:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet5[j]);
    }
      printf ("\nJNETSOL0:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet0[j]);
      }
      printf ("\n");
    }   
  }
  

  if (nohmm == 0 && nopsi == 1){
    
    if (printsty==0){
      printf("\nLength = %2d  Homologues = %2d\n",length,data[m]->numseq);
      printstring(letseq,letfilt,jury,alignlet,hmmlet,psi1let,psi2let,confidence,len,1,sollet25,sollet5,sollet0);
    }
    
    
    if (printsty==1){
      printf ("START PRED\n");
      for (j=0; j < length;j++){
	printf ("%c %c | %c %c %c %c %1.0f %5.5f %5.5f %5.5f\n",letseq[j],letfilt[j],alignlet[j],hmmlet[j],jury[j],sollet25[i],confidence[j],hmmnet[0][j],hmmnet[1][j],hmmnet[2][j]);
      }
      printf ("END PRED\n");
    }
    
  
    if (printsty==2){
      printf ("\njnetpred:");
      for (j=0; j < length;j++){
	printf ("%c,",letfilt[j]);
      } 
      
      printf ("\nJNETPROPH:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",hmmnet[0][j]);
      } 
      printf ("\nJNETPROPB:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",hmmnet[1][j]);
      } 
      printf ("\nJNETPROPC:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",hmmnet[2][j]);
      }
      printf ("\nJNETCONF:");
      for (j=0; j < length;j++){
	printf ("%1.0f,",confidence[j]);
      }
      printf ("\nJNETSOL25:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet25[j]);
      }
      printf ("\nJNETSOL5:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet5[j]);
      }
      printf ("\nJNETSOL0:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet0[j]);
      }
      printf ("\nJNETHMM:");
      for (j=0; j < length;j++){
        printf ("%c,",hmmlet[j]);
      }
      printf ("\nJNETALIGN:");
      for (j=0; j < length;j++){
        printf ("%c,",alignlet[j]);
      }     
      printf ("\n");
  }  


    if (printsty ==3){
      printf ("<HTML><BODY BGCOLOR=#ffffff><PRE>\n\n");	
      printf ("\nJnet:");
      for (j=0; j < length;j++){
	if (letfilt[j] == 'H'){
	  printf ("<font color=#ff0000>%c</font>",letfilt[j]);
	}
	if (letfilt[j] == 'E'){
	  printf ("<font color=#00ff00>%c</font>",letfilt[j]);
        }
	if (letfilt[j] == '-'){
	  printf ("<font color=#000000>%c</font>",letfilt[j]);
        } 
      } 
      printf ("</PRE></BODY></HTML>");
    }
  }
  
  
  if (nohmm == 0 && nopsi ==0){
    if (printsty==0){
      printf("\nLength = %2d  Homologues = %2d\n",length,data[m]->numseq);
      printstring(letseq,letfilt,jury,alignlet,hmmlet,psi1let,psi2let,confidence,len,2,sollet25,sollet5,sollet0);
    }
    
    
    if (printsty==1){
      printf ("START PRED\n");
      for (j=0; j < length;j++){
	printf ("%c %c | %c %c %c %c %c %1.0f %5.5f %5.5f %5.5f\n",letseq[j],letfilt[j],alignlet[j],hmmlet[j],psi2let[j],psi1let[j],jury[j],confidence[j],psi2net[0][j],psi2net[1][j],psi2net[2][j]);
      }
      printf ("END PRED\n");
    }
    
    
    if (printsty==2){
      printf ("\njnetpred:");
      for (j=0; j < length;j++){
	printf ("%c,",letfilt[j]);
      } 
      printf ("\nJNETPROPH:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",psi2net[0][j]);
      } 
      printf ("\nJNETPROPB:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",psi2net[1][j]);
      } 
      printf ("\nJNETPROPC:");
      for (j=0; j < length;j++){
	printf ("%5.5f,",psi2net[2][j]);
      }
      printf ("\nJNETCONF:");
      for (j=0; j < length;j++){
	printf ("%1.0f,",confidence[j]);
      }
      printf ("\nJNETHMM:");
      for (j=0; j < length;j++){
        printf ("%c,",hmmlet[j]);
      }
      printf ("\nJNETALIGN:");
      for (j=0; j < length;j++){
        printf ("%c,",alignlet[j]);
      }     
      printf ("\nJNETPSSM:");
      for (j=0; j < length;j++){
	printf ("%c,",psi2let[j]);
      }      
      printf ("\nJNETFREQ:");
      for (j=0; j < length;j++){                                
	printf ("%c,",psi1let[j]);                                
      } 
      printf ("\nJNETJURY:");
      for (j=0; j < length;j++){
	printf ("%c,",jury[j]);
      } 
      printf ("\nJNETSOL25:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet25[j]);
      }
      printf ("\nJNETSOL5:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet5[j]);
      }
      printf ("\nJNETSOL0:");
      for (j=0; j < length;j++){
	printf ("%c,",sollet0[j]);
      }
      printf ("\n");
    }   
  }
}


float doconf (float confa, float confb, float confc){
  
  float whichout;
  float maxout;
  float maxnext;
  float outconf;

  maxout=whichout=outconf=maxnext=0;
  
  maxout=confc;
  
  if (confa > confb && confa > confb  ){
    whichout=0;
    maxout=confa;
  }
  if (confb > confa && confb > confc  ){
    whichout=1;
    maxout=confb;
  }
  if (confc > confa && confc > confb  ){
    whichout=2;
    maxout=confc;
  }      
  if (whichout == 0){
    if (confb > confc){
      maxnext=confb;
    }
    if (confc > confb){
      maxnext=confc;
    }
  }     
  if (whichout == 1){
    if (confc > confa){
      maxnext=confc;
    }
    if (confa > confc){
      maxnext=confa;
    }
  }      
  if (whichout == 2){
    if (confb > confa){
      maxnext=confb;
    }
    if (confa > confb){
      maxnext=confa;
    }
  }
  outconf = (10*(maxout-maxnext));
  if (outconf > 9){
    outconf = 9;
  }
  return outconf;
}


void filter (char inp[MAXSEQLEN], char outp[MAXSEQLEN])

{
  char tmpstr[MAXSEQLEN];
  char replstr[10];
  char replstr2[10];

  strcpy(replstr,  "EHHHE");
  strcpy(replstr2, "EEEEE");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr,  "-HHH-");
  strcpy(replstr2, "HHHHH");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr,  "EHHH-");
  strcpy(replstr2, "EHHHH");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr,  "HHHE-");
  strcpy(replstr2, "HHH--");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr,  "-EHHH");
  strcpy(replstr2, "--HHH");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "EHHE");
  strcpy(replstr2, "EEEE");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "-HEH-");
  strcpy(replstr2, "-HHH-");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "EHH-");
  strcpy(replstr2, "EEE-");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);
 
  strcpy(replstr, "-HHE");
  strcpy(replstr2, "-EEE");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "-HH-");
  strcpy(replstr2, "----");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "HEEH");
  strcpy(replstr2, "EEEE");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "-HE");
  strcpy(replstr2, "--E");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "EH-");
  strcpy(replstr2, "E--");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);
  
  strcpy(replstr, "-H-");
  strcpy(replstr2, "---");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);
  
  strcpy(replstr, "HEH");
  strcpy(replstr2, "HHH");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "-E-E-");
  strcpy(replstr2, "-EEE-");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "E-E");
  strcpy(replstr2, "EEE");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "H-H");
  strcpy(replstr2, "HHH");
  StrReplStr(outp,inp,replstr,replstr2);
  strcpy(inp,outp);

  strcpy(replstr, "EHE");
  strcpy(replstr2, "EEE");
  StrReplStr(outp,inp,replstr,replstr2); 
  strcpy(inp,outp);
}



void StrReplStr(char *s1,char *s2,char *FromSStr, char *ToSStr)
{
char *ChP1,*ChP2;

s1[0]='\0';
ChP1=s2;


while ((ChP2=strstr(ChP1,FromSStr))!=NULL)
        {
        if (ChP1 != ChP2)
             Strncat(s1,ChP1,strlen(ChP1)-strlen(ChP2));
        strcat(s1,ToSStr);
        ChP1=ChP2+strlen(FromSStr);
        }
strcat(s1,ChP1);
return;
}


char *Strncat(char *s1,char *s2,int len)
{
  int OrigLen=0;
  if (len == 0)
    {
      fprintf (stderr ,"Strncat error!");
      return s1;
    }
  
  if (s1 == NULL || s2 == NULL)
    {
      fprintf (stderr, "Strncat error!");
      return NULL;
    }
  OrigLen=strlen(s1);
  if (strncat(s1,s2,len)==NULL)
    {
      fprintf (stderr, "Strncat error!");
      return NULL;
    }
  
  s1[OrigLen+len]='\0';
  
  return s1;
}


void doprofwin(alldata *data[], int arlen, int win, int curpos, int winar2[30][30],int m, int whichmat)
{
  int i,j,k;
  float addon;
  
  j=k=i=0;
  
    
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){     
    for (k=0;k < 24;k++){
      winar2[j][k]=0;
      
      
      if (whichmat == 0){
	winar2[j][k]=data[m]->profmat[i][k];
      }
      if (whichmat == 1){
	winar2[j][k]=data[m]->profile[i][k];
      }
    }
    
    if (i >= 0 && i < arlen ){ 
      addon = ((data[m]->conserv[i])*10);
     
	  winar2[j][24]=addon;
    }
    if (i < 0 ){
      for (k=0;k < 25;k++){
	winar2[j][k]=0;
      }
    }   
    if (i >= arlen ){ 
      for (k=0;k < 25;k++){
	winar2[j][k]=0; 
      }
    }  
  
    j++;
  }
}

void doprofpsi (alldata *data[], int arlen, int win, int curpos, float psiar[30][30],int m, int whichmat)
{
  int i,j,k;
  float addon;

  j=k=i=0;
  
  
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){     
    for (k=0;k < 20;k++){
      psiar[j][k]=0;
	psiar[j][k]=data[m]->psimat[i][k];
	
    }
   
    if (i < 0 ){
      for (k=0;k < 20;k++){
	psiar[j][k]=0;
      }
    }   
    if (i >= arlen ){ 
      for (k=0;k < 20;k++){
	psiar[j][k]=0; 
      }
    }    
    j++;
  }
}

void doprofpsi2 (alldata *data[], int arlen, int win, int curpos, int winar2[30][30],int m, int whichmat)
{
  int i,j,k;
  float addon;
  j=k=i=0;
  
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){     
    for (k=0;k < 20;k++){
      winar2[j][k]=0;

	winar2[j][k]=data[m]->psimat2[i][k];     
    }
  
    if (i < 0 ){
      for (k=0;k < 20;k++){
	winar2[j][k]=0;
      }
    }   
    if (i >= arlen ){ 
      for (k=0;k < 20;k++){
	winar2[j][k]=0; 
      }
    }    
    j++;
  }
}

void printstring (char *seqdef, 
		  char pred[MAXSEQLEN], 
		  char jury[MAXSEQLEN],
		  char alignlet[MAXSEQLEN],
		  char hmmlet[MAXSEQLEN],
		  char psi1let[MAXSEQLEN],
		  char psi2let[MAXSEQLEN],
		  float confidence[MAXSEQLEN], 
		  int len, int predmode,
	          char sollet25[MAXSEQLEN],
		  char sollet5[MAXSEQLEN],
		  char sollet0[MAXSEQLEN]
			)

{
  int i,j,y,chunks,cur;
  int outconf;
  char *res;
  chunks=0;
  outconf=0;
	
  FILE *ofp;
  char outputFilename[] = "alignment.txt";
  ofp = fopen(outputFilename, "w");
  if (ofp == NULL) {
	fprintf(stderr, "Can't open output file %s!\n",
			outputFilename);
	exit( 1 );
  }
	
  
  for (i=0; i < strlen(pred);i++){
    if ((i % len) == 0 && i != 0){
      chunks++;
    }
  }
  cur=0;
  
  
  for (y=0;y<=chunks;y++){  
    //printf( " RES\t: ");
    for (i=cur;i < (cur+len);i++){
      if(i < strlen(pred) ){
	//printf("%c",*seqdef);
	seqdef++; 
      }
    }    
    //printf("\n");
    if (predmode < 1){
      //printf(" ALIGN\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",alignlet[i]);
	}
      } //printf("\n");
    }
 
    if (predmode == 1){
      //printf(" ALIGN\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",alignlet[i]);
	}
      } //printf("\n");
      //printf(" HMM\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",hmmlet[i]);
	}
      } //printf("\n");
    }

    if (predmode == 2){
      //printf(" ALIGN\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",alignlet[i]);
	}
      } //printf("\n");
      //printf(" HMM\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",hmmlet[i]);
	}
      } //printf("\n");
      //printf(" FREQ\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",psi1let[i]);
	}
      } //printf("\n");
      
      //printf(" PSSM\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  //printf("%c",psi2let[i]);
	}
      } //printf("\n");
    }    
    
    //printf(" CONF\t: ");
    for (i=cur;i<(cur+len);i++){
      if(i < strlen(pred) ){
	outconf= confidence[i];
	//printf("%1d",outconf);
      }
    }
    
    
    if (predmode == 2){
      //printf("\n");
      //printf(" NOJURY\t: ");
      for (i=cur;i<(cur+len);i++){
	if(i < strlen(pred) ){
	  if (jury[i] == '<'){jury[i]='*';}
	  //printf("%c",jury[i]);
	}
      }
    }
          
    //printf("\n");
    fprintf(ofp,"FINAL\t: ");  
    for (i=cur;i<(cur+len);i++){
      if(i < strlen(pred) ){
	fprintf(ofp,"%c",pred[i]);
      }
    }
    fprintf(ofp,"\n");
    //printf(" SOL25\t: ");
    for (i=cur;i<(cur+len);i++){
      if(i < strlen(pred) ){
	//printf("%c",sollet25[i]);
      }
    }  
    //printf("\n");
    //printf(" SOL5\t: ");
    for (i=cur;i<(cur+len);i++){
      if(i < strlen(pred) ){
	//printf("%c",sollet5[i]);
      }
    }  
    //printf("\n");
    //printf(" SOL0\t: ");
    for (i=cur;i<(cur+len);i++){
      if(i < strlen(pred) ){
	//printf("%c",sollet0[i]);
      }
    }  
    
    //printf("\n");
    cur=cur+len;
    //printf("\n\n");
  }  
	fclose(ofp);
}




float check_acc (int *secdef, int length, int pred[MAXSEQLEN])
{
  int l;
  float acc;
  float acc3; 
  acc3=0;
  acc=0;
  
  for(l=0;l < length;l++){
    
    if (*secdef == pred[l]){
      acc3++;
    }
    secdef++;
  }
  acc=acc3/(float)(length);
  acc=acc*100;
  return (acc);
}



void getone(FILE *fsec, int seq[MAXSEQLEN],int *lenseq)
     
{
  char title[1000]={0};
  int c; 
  char letseq[MAXSEQLEN];
  int r;
  int lenr=0;
  *lenseq=0;

  c=r=0; 
  
  while((c=getc(fsec)) != '\n' && c != EOF ){
    title[r] = c;
    r++;
  }

  while ((c=getc(fsec)) != '>' && c != EOF){
    if (c != '\n' && c != ' ' && c != EOF){
      letseq[(*lenseq)++] = c;
      
    }
  }
  letseq[(*lenseq)] = 'n';
  seq2int(letseq,seq,*lenseq); 
  ungetc (c,fsec); 
}

void readhmm(FILE *hmmfile, alldata *data[]){

  char c;
  int x,i,y;
  x=i=y=0;
  while ((getc(hmmfile)) != EOF){

      for (i=0;i<24;i++){
        fscanf(hmmfile,"%f",&data[0]->hmmmat[x][i]);
      }
      x++;
  }
}



void int2seq (int seq[MAXSEQLEN],char letseq[MAXSEQLEN], int length)
{
    int i;
   
    for (i=0;i<=length;i++){
      if (seq[i] == 0  ) letseq[i]='A';
      if (seq[i] == 1  ) letseq[i]='R';
      if (seq[i] == 2  ) letseq[i]='N';
      if (seq[i] == 3  ) letseq[i]='D';
      if (seq[i] == 4  ) letseq[i]='C';
      if (seq[i] == 5  ) letseq[i]='Q';
      if (seq[i] == 6  ) letseq[i]='E';
      if (seq[i] == 7  ) letseq[i]='G';
      if (seq[i] == 8  ) letseq[i]='H';
      if (seq[i] == 9  ) letseq[i]='I';
      if (seq[i] == 10 ) letseq[i]='L';
      if (seq[i] == 11 ) letseq[i]='K';
      if (seq[i] == 12 ) letseq[i]='M';
      if (seq[i] == 13 ) letseq[i]='F';
      if (seq[i] == 14 ) letseq[i]='P';
      if (seq[i] == 15 ) letseq[i]='S';
      if (seq[i] == 16 ) letseq[i]='T';
      if (seq[i] == 17 ) letseq[i]='W';
      if (seq[i] == 18 ) letseq[i]='Y';
      if (seq[i] == 19 ) letseq[i]='V';
      if (seq[i] == 20 ) letseq[i]='B';
      if (seq[i] == 21 ) letseq[i]='Z';
      if (seq[i] == 22 ) letseq[i]='X';
      if (seq[i] == 23 ) letseq[i]='.';
      if (seq[i] == 25 ) letseq[i]='\0';

    } 
}


void seq2int(char letseq[MAXSEQLEN],int seq[MAXSEQLEN], int length)

{
int i;

 for (i=0;i<=length;i++){
   
   if (letseq[i] == 'A') seq[i]=0;
   if (letseq[i] == 'R') seq[i]=1;
   if (letseq[i] == 'N') seq[i]=2;
   if (letseq[i] == 'D') seq[i]=3;
   if (letseq[i] == 'C') seq[i]=4;
   if (letseq[i] == 'Q') seq[i]=5;
   if (letseq[i] == 'E') seq[i]=6;
   if (letseq[i] == 'G') seq[i]=7;
   if (letseq[i] == 'H') seq[i]=8;
   if (letseq[i] == 'I') seq[i]=9;
   if (letseq[i] == 'L') seq[i]=10;
   if (letseq[i] == 'K') seq[i]=11;
   if (letseq[i] == 'M') seq[i]=12;
   if (letseq[i] == 'F') seq[i]=13;
   if (letseq[i] == 'P') seq[i]=14;
   if (letseq[i] == 'S') seq[i]=15;
   if (letseq[i] == 'T') seq[i]=16;
   if (letseq[i] == 'W') seq[i]=17;
   if (letseq[i] == 'Y') seq[i]=18;
   if (letseq[i] == 'V') seq[i]=19;
   if (letseq[i] == 'B') seq[i]=20; 
   if (letseq[i] == 'Z') seq[i]=21;
   if (letseq[i] == 'X') seq[i]=22;
   if (letseq[i] == '.') seq[i]=23;
   if (letseq[i] == 'n') seq[i]=25; 

 }

}



void readpsi(FILE *psifile, alldata *data[]){

  char c;
  int x,i,y;
  x=i=y=0;
  while ((getc(psifile)) != EOF){
    

      for (i=0;i<20;i++){
	fscanf(psifile,"%f",&data[0]->psimat[x][i]);

      }
      x++;
  }
}

void readpsi2(FILE *psifile2, alldata *data[]){


  char c;
  int x,i,y;
  x=i=y=0;
  while ((getc(psifile2)) != EOF){

    for (i=0;i<20;i++){
      fscanf(psifile2,"%2d",&data[0]->psimat2[x][i]);
    }
    x++;  
  }
}

void int2sec(int sec[MAXSEQLEN], char letsec[MAXSEQLEN], int length)
{

int i;

 for (i=0;i<=length;i++){

   if (sec[i] == 1) letsec[i]='H';     
   if (sec[i] == 2) letsec[i]='E';     
   if (sec[i] == 3) letsec[i]='-'; 
   if (sec[i] == 25) letsec[i]='\0'; 
 }
}
void int2pred(int pred[MAXSEQLEN], char letpred[MAXSEQLEN], int length)
{

int i;

 for (i=0;i<=length;i++){
   if (pred[i] == 1) letpred[i]='H';     
   if (pred[i] == 2) letpred[i]='E';     
   if (pred[i] == 3) letpred[i]='-'; 
   if (pred[i] == 25){letpred[i]='\0';} 
 }
}

void pred2int(int pred[MAXSEQLEN], char letpred[MAXSEQLEN], int length)

{

int i;

 for (i=0;i<=length;i++){
   if (letpred[i] == 'H') pred[i]=1;     
   if (letpred[i] == 'E') pred[i]=2;     
   if (letpred[i] == '-') pred[i]=3; 
   if (letpred[i] == '\0') pred[i]=25; 

 }
}


void doprofhmm(alldata *data[], int arlen, int win, int curpos, float psiar[30][30],int m, int whichmat)
{
  int i,j,k;
  float addon;

  j=k=i=0;
  
  for (i=(curpos-((win-1)/2)); i <=(curpos+((win-1)/2)) ; i++){     
    for (k=0;k < 24;k++){
      psiar[j][k]=0;
	psiar[j][k]=data[m]->hmmmat[i][k]; 	
    }
    if (i < 0 ){
      for (k=0;k < 24;k++){
	psiar[j][k]=0;
      }
    }   
    if (i >= arlen ){ 
      for (k=0;k < 24;k++){
	psiar[j][k]=0; 
      }
    }    
    j++;
  }
}
