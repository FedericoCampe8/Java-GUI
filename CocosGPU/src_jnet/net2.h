/*********************************************************
  net2.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:30 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int net2(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} net2REC = {76,3,net2};
