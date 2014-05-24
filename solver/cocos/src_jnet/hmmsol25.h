/*********************************************************
  hmmsol25.h
  --------------------------------------------------------
  generated at Mon Sep 13 11:07:12 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int hmmsol25(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} hmmsol25REC = {408,2,hmmsol25};
