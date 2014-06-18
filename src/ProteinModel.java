package jafatt;

import java.util.ArrayList;
import java.util.List;
import java.util.Observable;
import org.biojava.bio.seq.Sequence;
import org.biojava.bio.structure.AminoAcid;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;

public class ProteinModel extends Observable{
    
    protected UserFrame view;
    
    /* Current Target */
    private String currentTargetString;
    private Sequence currentTargetSequence;
    private String targetPath;
    private boolean isPDB;
    
    /* Current protein */
    private Structure currentStructure;
    private String currentStructurePath;
    private String currentStructureString;
    private int proteinResiduesOffset;
    
    protected String idProteinCode;
    protected int totalLoadedProteins;
    
    /* Current fragment */
    private Fragment currentExtractionFragment;
    private Fragment currentAssemblingFragment;
    
    /* Current fragment */
    private Fragment pastAssemblingFragment;
    
    /* Deleted fragment */
    private Fragment deletedExtractionFragment;
    private Fragment deletedAssemblingFragment;
    private Fragment deselectedAssemblingFragment;
    
    /* Proteins loaded */
    private ArrayList<String> proteinsLoaded;
    
    /* Fragments present on Extraction panel */
    private ArrayList<Fragment> frgsExtraction;
    
    /* Fragments present on Assembling panel */
    private ArrayList<Fragment> frgsAssembling;
    
    private ArrayList<Fragment> frgsSelectedA;
    
    public ProteinModel(){
        setup();
    }
    
    /* Setup */
    private void setup(){
        totalLoadedProteins = 0;
        currentExtractionFragment = null;
        currentAssemblingFragment = null;
        pastAssemblingFragment = null;
        deselectedAssemblingFragment = null;
        proteinsLoaded = new ArrayList<String>();
        frgsExtraction = new ArrayList<Fragment>();
        frgsAssembling = new ArrayList<Fragment>();
        frgsSelectedA = new ArrayList<Fragment>();
    }//setup
    
    /* Set view */
    public void setView( UserFrame view ){
        this.view = view;
        addObserver(view);
    }//setView
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           GENERAL METHODS             *
     *                                       *
     *****************************************   
     *****************************************/
    
    
    
    /* Delete a fragment */
    public void deleteFragment(int i, int panel){
        if(panel == Defs.EXTRACTION)
            deleteFragmentE(i);
        else
            deleteFragmentA(i);
    }//deleteFragment
    
    /* Set the current fragment */
    public void setCurrentFragment(Fragment fragment, int panel){
        if(panel == Defs.EXTRACTION)
            currentExtractionFragment = fragment;
        else
            currentAssemblingFragment = fragment;      
    }//setFragmentOffset
    
    /* Get the current fragment */
    public Fragment getCurrentFragment(int panel){
        if(panel == Defs.EXTRACTION)
            return currentExtractionFragment;
        else
            return currentExtractionFragment;
    }//getCurrentFragment
    
    /* Get the deleted fragment */
    public Fragment getDeletedFragment(int panel){
        if(panel == Defs.EXTRACTION)
            return deletedExtractionFragment;
        else
            return deletedAssemblingFragment;
    }//getDeletedFragment
    
    /* Get all the fragments present on Extraction panel */
    public ArrayList<Fragment> getAllFragments(int panel){
        if(panel == Defs.EXTRACTION)
            return frgsExtraction;
        if(panel == Defs.ASSEMBLING)
            return frgsAssembling;
        else
            return frgsSelectedA;
    }//getAllFragments
    
    /* Check if all the fragments on Extraction are already present in the 
     * Assembling panel
     */
    public boolean fragmentsAlreadyPresent(){
        boolean ok = true;
        int firstEAA = 0;
        int lastEAA = -1;
        int firstAAA = 0;
        int lastAAA = -1;
        
        for(int i = 0; i < frgsExtraction.size(); i++){
            Fragment eFrg = frgsExtraction.get(i);
            String eID = eFrg.getParameters()[Defs.FRAGMENT_PROTEIN_ID];
            try{
                firstEAA = Integer.parseInt(eFrg.getParameters()[Defs.FRAGMENT_START_AA]);
                lastEAA = Integer.parseInt(eFrg.getParameters()[Defs.FRAGMENT_END_AA]);
            }catch(NumberFormatException nfe){}
            
            /* Check with the assembling's fragments */
            for(int j = 0; j < frgsAssembling.size(); j++){
                Fragment aFrg = frgsAssembling.get(j);
                
                if(eID.equals(aFrg.getParameters()[Defs.FRAGMENT_PROTEIN_ID])){
                    try{
                        firstAAA = Integer.parseInt(aFrg.getParameters()[Defs.FRAGMENT_START_AA]);
                        lastAAA = Integer.parseInt(aFrg.getParameters()[Defs.FRAGMENT_END_AA]);
                    }catch(NumberFormatException nfe){}
                    
                    //if(firstEAA >= firstAAA || lastEAA <= lastAAA){
                    if(firstEAA == firstAAA && lastEAA == lastAAA){
                        ok = false;
                        return ok;
                    }        
                }
            }
        }
        
        return ok;
    }//fragmentsAlreadyPresent
    
    /* Inform the Observers */
    protected void inform(Object info){
        
        /* Set changes */
        setChanged();
        
        /* Notify the observers */
        notifyObservers(info);
    }//inform
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           TARGET METHODS              *
     *                                       *
     *****************************************   
     *****************************************/
    
    public void setTargetPath(String targetPath, boolean isPDB){
        this.targetPath = targetPath;
        this.isPDB = isPDB;   
    }
    
    public String getTargetPath(){
        if(!isPDB)
            return targetPath;
        return "";
    }
    
    /* Set target's string */
    public void setTargetString( String target ){
        currentTargetString = target;
    }//setTargetString
    
    /* set target structure */
    public void setTarget( Sequence target ){
        currentTargetSequence = target;
        
        /* Inform observers */
        inform(Defs.NEW_TARGET);
    }//setTarget
    
    /* Get Target sequence */
    public Sequence getTargetSequence(){
        return currentTargetSequence;
    }//getTargetSequence
    
    /* Get Target string */
    public String getTargetSequenceString(){
        if(!currentTargetString.isEmpty())
            return currentTargetString;
        else
            return "";
    }//getTargetSequenceString
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           PROTEIN METHODS             *
     *                                       *
     *****************************************   
     *****************************************/
    
    
    
    /* Set the current loaded structure */
    public boolean setCurrentProtein( Structure proteinStructure, String proteinPath ){
        String seq = null;
        
        /* Chain made by residues */
        Chain chain = null;
        currentStructure = proteinStructure;
        currentStructurePath = proteinPath;
        //idProteinCode = currentStructure.getPDBCode();
        idProteinCode = HeaderPdb.getProteinId();
        //idProteinCode = currentStructure.getResidueNumber();  /* v 3.0 */
        
        /* Set the sequence */
        try{
            chain = currentStructure.getChain(0);
            seq = chain.getSeqResSequence();
        }catch(Exception e){
            return false;
        }
        
        /* Add a new sequence */
        proteinsLoaded.add(seq);
        
        /* Check the offset */
        proteinResiduesOffset = setProteinOffset();
        System.out.println("proteinResiduesOffset " + proteinResiduesOffset);
        
        /* Set the structure string */
        currentStructureString = setProteinString();
        System.out.println("currentStructureString " + currentStructureString);
        
        totalLoadedProteins++;
        
        /* Inform the observers */
        inform(Defs.NEW_PROTEIN);
        
        /* Return */
        return true;
    }//setCurrentStructure
    
    /* Get the current structure */
    public Structure getCurrentProtein(){
        return currentStructure;
    }//getCurrentProtein
    
    /* Get the current protein's path */
    public String getCurrentPath(){
        return currentStructurePath;
    }//getCurrentPath
    
    /* Set the protein sequence string */
    private String setProteinString(){
        //return currentStructure.getChain(0).getSeqResSequence();
        return HeaderPdb.getTargetSequence();
    }//setProteinString
    
    /* Get the current protein sequence */
    public String getCurrentProteinSequenceString(){
        return currentStructureString;
    }//getCurrentProteinSequence
    
    /* Set the actual protein offset */
    private int setProteinOffset(){
        
        /* Set the sequence of the loaded protein */
        Chain chain = currentStructure.getChain(0);
        
        /* Check the offset of the loaded protein (i.e., the first AA position) */
        List groups = chain.getAtomGroups("amino");
        
        /* First AA */
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        
        /* Set the offset */
        return Utilities.getAAPosition(firstAA);       
    }//setProteinOffset
    
    /* Get the current protein offset */
    public int getCurrentProteinOffset(){
        return proteinResiduesOffset;
    }//getCurrentProteinOffset
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           EXTRACTION METHODS          *
     *                                       *
     *****************************************   
     *****************************************/
    
    
    
    /* Add a new fragment on the Extraction panel*/
    public void addFragmentE(String info1[], String info2[]){
        /* Create a new fragment */
        Fragment frg = new Fragment(info1, info2, this);
        frgsExtraction.add(frg);
        currentExtractionFragment = frg;
        
        /* Debug */
        /*System.out.println("RES1 " + frg.getParameters()[Defs.FRAGMENT_START_AA] + " RES2 " +
                frg.getParameters()[Defs.FRAGMENT_END_AA]); */
        
        /* Inform the observer of a new fragment */
        inform(Defs.ADD_FRAGMENT_ON_EXTRACTION);
    }//addFragmentOnExtraction
    
    /* Delete a fragment */
    public void deleteFragmentE(int i){
        
        /* Set the deleted fragment for inform the View on about the fragment 
         * to color
         */
        deletedExtractionFragment = frgsExtraction.get(i);
        
        /* Remove the fragment */
        frgsExtraction.remove(i);
        
        /* Set the most recent fragment as current fragment */
        if(frgsExtraction.size() > 0)
            currentExtractionFragment = getMostRecentFragmentE();
        
        /* Inform the observer */
        inform(Defs.DELETE_FRAGMENT);
    }//deleteFragment
    
    /* Delete all fragments */
    public void deleteFragmentE(){
        frgsExtraction.clear();
        inform(Defs.DELETE_FRAGMENT);
    }//deleteFragment
    
    /* Set the current fragment */
    public void setCurrentFragmentE(Fragment fragment){
            currentExtractionFragment = fragment;     
    }//setFragmentOffset
    
    /* Get the current fragment */
    public Fragment getCurrentFragmentE(){
            return currentExtractionFragment;
    }//getCurrentFragment
    
    /* Get the deleted fragment */
    public Fragment getDeletedFragmentE(){
            return deletedExtractionFragment;
    }//getDeletedFragment
    
    /* Get all the fragments present on Extraction panel */
    public ArrayList<Fragment> getAllFragmentsE(){
            return frgsExtraction;
    }//getAllFragments
    
    /* Get the most recent fragment */
    private Fragment getMostRecentFragmentE(){
        long mostRecent = -1;
        int numFragments = frgsExtraction.size();
        Fragment mostRecentFragment = frgsExtraction.get(0);
        long fragmentTime = -1;

        /* Set the "base" time */
        try{
            fragmentTime = Long.parseLong(
                    frgsExtraction.get(0).getParameters()[Defs.FRAGMENT_TIME_OF_CREATION]
                    );
        }catch(NumberFormatException nfe){
            System.out.println("Error on parse Long Format in "
                    + "getMostRecentFragment() (1) method: " + nfe);
        }

        for(int i = 1; i < numFragments; i++){
            try{
                mostRecent = Long.parseLong(
                        frgsExtraction.get(i).getParameters()[Defs.FRAGMENT_TIME_OF_CREATION]
                        );
            }catch(NumberFormatException nfe){
                System.out.println("Error on parse Long Format in "
                    + "getMostRecentFragment() (2) method: " + nfe);
            }
            
            if(mostRecent > fragmentTime){
                fragmentTime = mostRecent;
                mostRecentFragment = frgsExtraction.get(i);
            }
        }

        return mostRecentFragment;
    }//getMostRecentFragmentE
    
    /* Verify if a fragment is present on the Extraction panel */
    public int isFragmentPresent(String info[]){
        for(int i = 0; i < frgsExtraction.size(); i++)
            if(frgsExtraction.get(i).verifyResidueIn(info))
                return i;
        return -1;
    }//isFragmentPresent
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           ASSEMBLING METHODS          *
     *                                       *
     *****************************************   
     *****************************************/
    
    
    
    /* Add a new fragment on the Assembling panel */
    public void addFragmentA(Fragment frg){
        
        /* Add a new fragment on Assembling panel */
        frgsAssembling.add(frg);
        currentAssemblingFragment = frg;
        
        /* Debug */
        /* System.out.println("RES1 " + frg.getParameters()[Defs.SAA] + " RES2 " +
                frg.getParameters()[Defs.EAA]); */
        
        /* Inform the observer of a new fragment */
        inform(Defs.SET_FRAGMENT_ON_ASSEMBLING);
        
    }//addFragmentA
    
    public void deleteFragmentA(int i){
    }//deleteFragmentA
    
    /* Delete all fragments */
    public void deleteFragmentA(){
        frgsAssembling.clear();
        frgsSelectedA.clear();
    }//deleteFragmentA
    
    /* Set the current fragment from info */
    /* DEPRECATED
    public void setCurrentFragmentA(String  numFragment){
        Fragment currentFragment = null;
        int numFragments = frgsAssembling.size();
        System.out.println("ASSEMBLING PANEL FRAGMENT SIZE " + numFragments);
        System.out.println("String numFragmnet " + numFragment);
        
        for(int i = 0; i < numFragments; i++){
            if(numFragment.equals(
                    frgsAssembling.get(i).getParameters()[Defs.FRAGMENT_NUM])
               ){
                currentFragment = frgsAssembling.get(i);
                break;
            }
        }
        
        if(currentFragment == null){
            view.printStringLn("Error on selecting the fragment into the "
                    + "ASSEMBLING panel");
            return;
        }
        
        setCurrentFragmentA(currentFragment);
    }//setFragmentOffset */
    
    
    public void selectFragmentA(String residue){
        Fragment currentFragment = null;
        int numFragments = frgsAssembling.size();
        
        int res = Integer.parseInt(residue);
        
        for(int i = 0; i < numFragments; i++){
            int start = Integer.parseInt(frgsAssembling.get(i).getParameters()[Defs.FRAGMENT_START_AA]);
            int end = Integer.parseInt(frgsAssembling.get(i).getParameters()[Defs.FRAGMENT_END_AA]);
            if((res >= start) && (res <= end)){
                currentFragment = frgsAssembling.get(i);
                break;
            }
        }
        /*
        if(currentFragment == null){
            view.printStringLn("Error on selecting the fragment into the "
                    + "ASSEMBLING panel");
            return;
        }*/
        
        selectFragmentA(currentFragment);
    }//setFragmentOffset */
    
    /* Set the current fragment */
    public void selectFragmentA(Fragment fragment){
        
        /* Check if there is a fragment on Assembling panel to set */
        if(frgsAssembling.isEmpty())
            return;
        
        currentAssemblingFragment = fragment;
        
        if(frgsSelectedA.indexOf(fragment) != -1){
            deselectFragmentA(frgsSelectedA.indexOf(fragment));
            return;
        }
        
        frgsSelectedA.add(fragment);
        
        /* Inform for the selected fragment 
        if(currentAssemblingFragment == null)
            inform(Defs.DESELECT_FRAGMENT_ON_ASSEMBLING);
        else
            inform(Defs.SELECT_FRAGMENT_ON_ASSEMBLING);*/
        inform(Defs.SELECT_FRAGMENT_ON_ASSEMBLING);
    }//setFragmentOffset
    
    /* Delete a fragment */
    private void deselectFragmentA(int i){
        
        deselectedAssemblingFragment = frgsSelectedA.get(i);
        
        /* Remove the fragment */
        frgsSelectedA.remove(i);
        
        /* Inform the observer */
        inform(Defs.DESELECT_FRAGMENT_ON_ASSEMBLING);
        
    }//deselectFragmentA
    
    public ArrayList<Fragment> getAllSelectedFragmentsA(){
            return frgsSelectedA;
    }//getAllFragments
    
    public Boolean isFragmentSelected(){
        return (currentAssemblingFragment == pastAssemblingFragment);
    }
    
    /* Get the current fragment */
    public Fragment getCurrentFragmentA(){
        return currentAssemblingFragment;
    }//getCurrentFragment
    
    public Fragment getDeselectedFragmentA(){
        return deselectedAssemblingFragment;
    }//getCurrentFragment
    
    /* Get the deleted fragment */
    public Fragment getDeletedFragmentA(){
        return deletedAssemblingFragment;
    }//getDeletedFragment
    
    /* Get all the fragments present on Assembling panel */
    public ArrayList<Fragment> getAllFragmentsA(){
        return frgsAssembling;
    }//getAllFragmentsA
    
    public int getDimensionFragmentsA(){
        return frgsAssembling.size();
    }//getAllFragmentsA
    
    /* Get the "successor" fragment or null otherwise, calculated
     * on the offset */
    public Fragment getNextFragment(Fragment currentFragment){
        int numFragments = frgsAssembling.size();
        Fragment nextFragment = null;
        Fragment tryFragment;
        
        String nOffset;
        int nextOffset;
        
        String cOffset = currentFragment.getParameters()[Defs.FRAGMENT_OFFSET];
        int currentLength = currentFragment.getParameters()[Defs.FRAGMENT_SEQUENCE_STRING].length();
        int currentOffset;
        int currentEnd;
        try{
            currentOffset = Integer.parseInt(cOffset);
        }catch(NumberFormatException nfe){
            view.printStringLn("Error in getNextFragment() (1): " + nfe);
            return null;
        }
        currentEnd = currentOffset + currentLength;
        
        int bestOffset = currentTargetString.length();
        /* Find the next fragment */
        for(int i = 0; i < numFragments; i++){
            tryFragment = frgsAssembling.get(i);
            nOffset = tryFragment.getParameters()[Defs.FRAGMENT_OFFSET];
            try{
                nextOffset = Integer.parseInt(nOffset);
            }catch(NumberFormatException nfe){
                view.printStringLn("Error in getNextFragment() (2): " + nfe);
                return null;
            }
            
            /* Find the "nearest" fragment */
            if(nextOffset > currentOffset && nextOffset < bestOffset){
                nextFragment = tryFragment;
                bestOffset = nextOffset;
            }
        }
        
        /* Return */
        return nextFragment;
    }//getNextFragment
    
    /* Get the groups of the "Block" constraint */
    public ArrayList<String[]> getBlockGroups(){
        ArrayList<String[]> groupsInfo = new ArrayList<String[]>();
        int numFragments = frgsAssembling.size();
        
        /* Check whether a fragment is part of any group */
        for(int i = 0; i < numFragments; i++){
            Fragment frg = frgsAssembling.get(i);
            
            /* In case, add the infos */
            if(frg.getConstraints()[Defs.CONSTRAINT_BLOCK]){
                String[] blockInfo = new String[2];
                
                /* Fragment number */
                blockInfo[Defs.CONSTRAINT_BLOCK_FRAGMENT_NUM] = 
                        frg.getParameters()[Defs.FRAGMENT_NUM];

                /* Fragment group */
                blockInfo[Defs.CONSTRAINT_BLOCK_GROUP] = frg.getGroup();
                
                /* Add infos */
                groupsInfo.add(blockInfo);
            }       
        }
        
        return groupsInfo;
    }//getBlockGroups
    
    /* Set a constraint on this fragment */
    public void setConstraintOnFragment(int constraintType, String[] infoConstraint){
        switch(constraintType){
            case Defs.CONSTRAINT_BLOCK:
                getCurrentFragmentA().setConstraint(constraintType, infoConstraint);
                inform(Defs.SET_CONSTRAINT_BLOCK);
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
    
}//proteinModel
