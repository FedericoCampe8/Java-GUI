/*********************************************************
  net1.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:27 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int net1(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} net1REC = {425,3,net1};
