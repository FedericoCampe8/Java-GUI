/*********************************************************
  psinet2c.h
  --------------------------------------------------------
  generated at Mon Aug  2 17:12:54 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int psinet2c(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} psinet2cREC = {57,3,psinet2c};
