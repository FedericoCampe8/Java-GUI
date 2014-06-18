/*********************************************************
  psinet2.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:52 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int psinet2(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} psinet2REC = {57,3,psinet2};
