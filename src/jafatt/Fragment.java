package jafatt;

public class Fragment{
    
    /* Protein model */
    private ProteinModel model;
    
    /* Id of the model where the fragment is extracted from */
    private String proteinID;
    
    /* Number of the loaded protein */
    private String fromProteinNum;
    
    /* Number of the extracted fragment from protein "proteinNum" */
    private String numFragment;
    
    /* General infos */
    private String startAtom;
    private String endAtom;
    private String startAA;
    private String endAA;
    private String startAAName;
    private String endAAName;
    private String fragmentStringSequence;
    private String offset;
    private String timeOfCreation;
    private String startAAP;
    private String endAAP;
    
    /* Constraints 
     * [BLOCK][VOLUME][EXACTD][WITHIND][THESE_COORDINATES]
     */
    private boolean[] constraints;
    private String blockGroup;
    
    public Fragment(String[] res1, String[] res2, ProteinModel model){
        int resNum1 = -1;
        int resNum2 = -1;
        int start = -1;
        int end = -1;
        
        this.model = model;
        
        /* Check the number of the fragment */
        if(res1[Defs.INFO_NUMPROT].equals(res2[Defs.INFO_NUMPROT])){
            numFragment = res1[Defs.INFO_NUMPROT];
            //System.out.println("NUM_FRAGMENT  ----------------- " + numFragment);
        }else{
            return;
        }
        try{
            resNum1 = Integer.parseInt(res1[Defs.INFO_RESNUM]);
            resNum2 = Integer.parseInt(res2[Defs.INFO_RESNUM]);
        }catch(NumberFormatException nfe){}
        
        /* Setting right infos */
        if(resNum1 <= resNum2){
            startAtom = res1[Defs.INFO_ATOM];
            endAtom = res2[Defs.INFO_ATOM];
            startAA = res1[Defs.INFO_RESNUM];
            endAA = res2[Defs.INFO_RESNUM];
            startAAName = res1[Defs.INFO_RESNAME];
            endAAName = res2[Defs.INFO_RESNAME];
            if(resNum1 == HeaderPdb.getFirstAA())
                start = resNum1;
            else
                start = resNum1 - 1;
            startAAP = Integer.toString(start);
            
            if(resNum2 == HeaderPdb.getLastAA())
                end = resNum2;
            else
                end = resNum2 + 1;
            endAAP = Integer.toString(end);
            
        }else{
            startAtom = res2[Defs.INFO_ATOM];
            endAtom = res1[Defs.INFO_ATOM];
            startAA = res2[Defs.INFO_RESNUM];
            endAA = res1[Defs.INFO_RESNUM];
            startAAName = res2[Defs.INFO_RESNAME];
            endAAName = res1[Defs.INFO_RESNAME];            
            if(resNum2 == HeaderPdb.getLastAA())
                start = resNum2;
            else
                start = resNum2 - 1;
            startAAP = Integer.toString(start);
            
            if(resNum1 == HeaderPdb.getLastAA())
                end = resNum1;
            else
                end = resNum1 + 1;
            endAAP = Integer.toString(end);
        }
        
        constraints = new boolean[Defs.CONSTRAINT_NUM];
        for(int i = 0; i < Defs.CONSTRAINT_NUM; i++)
            constraints[i] = false;
        
        proteinID = model.idProteinCode;

        fromProteinNum = Integer.toString(model.totalLoadedProteins);
        //offset = "";
        
        fragmentStringSequence = getFragmentSequence();
        
        timeOfCreation = "" + System.currentTimeMillis();
        
        /* Debug */
        /* System.out.println("Fragment sequence: " + fragmentStringSequence 
                            + " Time of creation: " + timeOfCreation); */
        
        /* Init items for constraints */
        blockGroup = Defs.EMPTY;
    }
    
    /* Dummy constructor, it after requires the clone method */
    public Fragment(ProteinModel model){
        this.model = model;
        constraints = new boolean[Defs.CONSTRAINT_NUM];
    }
                   
    /* Verify if a residue is contained in a fragment */
    public boolean verifyResidueIn(String[] info){
        int numProtein = -1;
        int startRes = -1;
        int endRes = -1;
        int myProtein = -1;
        int myRes = -1;
        try{
            startRes = Integer.parseInt(startAA);
            endRes = Integer.parseInt(endAA);
            myRes = Integer.parseInt(info[Defs.INFO_RESNUM]);
        }catch(NumberFormatException nfe){
            return true;
        }
        
        /* Return the check */
        return ( ( myRes >= startRes ) &&
                 ( myRes <= endRes )     
                );
    }//verifyResidueIn
    
    /* Return the parameters of the fragment */
    public String[] getParameters(){
        String[] parameters = new String[Defs.FRAGMENT_NUM_PARAMETERS];
        
        /* Set parameters */
        parameters[0] = proteinID;
        parameters[1] = fromProteinNum;
        parameters[2] = numFragment;
        parameters[3] = startAtom;
        parameters[4] = endAtom;
        parameters[5] = startAA;
        parameters[6] = endAA;
        parameters[7] = startAAName;
        parameters[8] = endAAName;
        parameters[9] = fragmentStringSequence;
        parameters[10] = offset;
        parameters[11] = timeOfCreation;
        parameters[12] = startAAP;
        parameters[13] = endAAP;
        /* Return parameters */
        return parameters;
    }//getParameters
    
    /* Clone the input fragment into this one */
    public void clone(Fragment source){
        setParameters(source.getParameters());
        
        /* Also clone the constraints 
         * TO DO
         */
        System.arraycopy(source.getConstraints(), 0, constraints, 
                0, Defs.CONSTRAINT_NUM);
        blockGroup = source.getGroup();
    }//clone
    
    /* Set the parameters from an external source */
    private void setParameters(String[] parameters){
        proteinID = parameters[Defs.FRAGMENT_PROTEIN_ID];
        fromProteinNum = parameters[Defs.FRAGMENT_PROTEIN_NUM];
        numFragment = parameters[Defs.FRAGMENT_NUM];
        startAtom = parameters[Defs.FRAGMENT_START_ATOM];
        endAtom = parameters[Defs.FRAGMENT_END_ATOM];
        startAA = parameters[Defs.FRAGMENT_START_AA];
        endAA = parameters[Defs.FRAGMENT_END_AA];
        startAAName = parameters[Defs.FRAGMENT_START_AA_NAME];
        endAAName = parameters[Defs.FRAGMENT_END_AA_NAME];
        fragmentStringSequence = parameters[Defs.FRAGMENT_SEQUENCE_STRING];
        offset = parameters[Defs.FRAGMENT_OFFSET];
        timeOfCreation = parameters[Defs.FRAGMENT_TIME_OF_CREATION];
        startAAP = parameters[Defs.FRAGMENT_RESNO_START];
        endAAP = parameters[Defs.FRAGMENT_RESNO_END];
    }//setParameters
    
    /* Set the offset with respect to the target */
    public void setOffset(String offset){
        this.offset = offset;
        
        /* Inform the observer about a new offset set on a fragment */
        model.inform(Defs.SET_OFFSET);
    }//setOffset
    
    /* Set the unique number of the fragment */
    public void setFragmentNumber(String numFragment){
        this.numFragment = numFragment;
    }//setFragmentNumber
    
    /* Get the fragment sequence string */
    private String getFragmentSequence(){
        int startAANum = -1;
        int endAANum = -1;
        
        try{
            startAANum = Integer.parseInt(startAA);
            endAANum = Integer.parseInt(endAA);
        }catch(NumberFormatException nfe){}
        
        /* Calculate the indexes of the substring */
        int beginIndex = startAANum - model.getCurrentProteinOffset();
        int endIndex = endAANum - model.getCurrentProteinOffset();
        
        /* Debug */
        /*
        System.out.println("Fragment start: " + startAANum + " end: " 
                + endAANum + " residue offset: " + model.getCurrentProteinOffset()
                + " beginIndex: " + beginIndex + " endIndex: " + endIndex
                + " current Structure String: " 
                + model.getCurrentProteinSequenceString());
         */
         
        /* Get the fragment string */
        return model.getCurrentProteinSequenceString().substring(beginIndex, endIndex);
    }//getFragmentSequence
    
    /* Set a constraint on this fragment */
    public void setConstraint(int constraint, String[] infoConstraint){
        switch(constraint){
            case Defs.CONSTRAINT_BLOCK:
                constraints[Defs.CONSTRAINT_BLOCK] = true;
                blockGroup = infoConstraint[Defs.CONSTRAINT_BLOCK_GROUP];
                break;
            case Defs.CONSTRAINT_VOLUME:
                break;
            case Defs.CONSTRAINT_EXACTD:
                break;
            case Defs.CONSTRAINT_WITHIND:
                break;
            case Defs.CONSTRAINT_THESE_COORDINATES:
                break;
        }
    }//setConstraint
    
    /* Return the constraints set on this fragment */
    public boolean[] getConstraints(){
        return constraints;
    }//getConstraints
    
    /* Return the group of the fragment */
    public String getGroup(){
        return blockGroup;
    }//getGroup
    
    /* Check if this fragments has some constraints imposed on it*/
    public boolean hasSomeConstraints(){
        int numConstraints = constraints.length;
        boolean con = false;
        
        for(int i = 0; i < numConstraints; i++)
            con = con || constraints[i];
        return con;
    }//hasSomeConstraints
    
}//Fragment
