/*********************************************************
  psinet2b.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:54 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int psinet2b(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} psinet2bREC = {57,3,psinet2b};
