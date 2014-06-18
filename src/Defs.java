package jafatt;

import java.awt.Color;

public class Defs {
    
    /* Panels */
    public static final int ASSEMBLING = 0;
    public static final int EXTRACTION = 1;
    public static final int TARGET = 2;
    public static final int INFOPANEL = 3;
    public static final int OUTPUT = 4;
    
    public static final int ASSEMBLING_S = 5;
    
    /* Colors */
    public static final Color LINECOLOR = new Color(185,185,185);
    public static final Color INNERCOLOR = new Color(240, 240, 240);
    public static final Color BACKGROUNDCOLOR = new Color(150, 150, 150);
    public static final Color BUTTONBACKGROUNDCOLOR = new Color(219,219,219);
    
    /* Target Panel */
    public static final int PRIMARY = 0;
    public static final int SECONDARY = 1;
    
    /* UserFrame Panel */
    public static final int NEW_TARGET = 0;
    public static final int NEW_PROTEIN = 1;
    public static final int DELETE_FRAGMENT = 2;
    public static final int ADD_FRAGMENT_ON_EXTRACTION = 3;
    public static final int SET_FRAGMENT_ON_ASSEMBLING = 4;
    public static final int SET_OFFSET = 5;
    public static final int SELECT_FRAGMENT_ON_ASSEMBLING = 6;
    public static final int DESELECT_FRAGMENT_ON_ASSEMBLING = 7;
    public static final int SET_CONSTRAINT_BLOCK = 8;
    public static final int DEL_ALL_FRG_A = 9;
    
    /* Controller */
    public static final int INFO_NUMPROT = 0;
    public static final int INFO_RESNAME = 1;
    public static final int INFO_RESNUM = 2;
    public static final int INFO_ATOM = 3;
    public static final int INFO_AX = 4;
    public static final int INFO_AY = 5;
    public static final int INFO_AZ = 6;
    
    /* Target panel */
    public static final int INITIALWRAP = 5000;
    
    /* Extraction panel */
    public static final String DUMMY_PDB_FILE = "transferPDBFile.pdb";
    //public static final String IN_PDB_FILE = "PDBFile.pdb";
    
    /* Assembling panel */
    public static final int CONSTRAINT_NUM = 5;
    
    public static final int CONSTRAINT_BLOCK = 0;
    public static final int CONSTRAINT_VOLUME = 1;
    public static final int CONSTRAINT_EXACTD = 2;
    public static final int CONSTRAINT_WITHIND = 3;
    public static final int CONSTRAINT_THESE_COORDINATES = 4;
    
    /* Constraint Block */
    public static final int CONSTRAINT_BLOCK_FRAGMENT_NUM = 0;
    public static final int CONSTRAINT_BLOCK_GROUP = 1;
    public static final String[] GROUP_COLORS = {"red", "blue", "cyan", "yellow", 
        "gold", "turquoise", "deeppink", "chartreuse", "mediumspringgreen",
        "aquamarine"};
    public static final int GROUP_COLORS_NUM = GROUP_COLORS.length;
    
    /* Fragment */
    public static final String EMPTY = "0";
    public static final int FRAGMENT_NUM_PARAMETERS = 12;
    
    /***********************************************************/
    
    public static final int FRAGMENT_PROTEIN_ID = 0;
    public static final int FRAGMENT_PROTEIN_NUM = 1;
    public static final int FRAGMENT_NUM = 2;
    public static final int FRAGMENT_START_ATOM = 3;
    public static final int FRAGMENT_END_ATOM = 4;
    public static final int FRAGMENT_START_AA = 5;
    public static final int FRAGMENT_END_AA = 6;
    public static final int FRAGMENT_START_AA_NAME = 7;
    public static final int FRAGMENT_END_AA_NAME = 8;
    public static final int FRAGMENT_SEQUENCE_STRING = 9;
    public static final int FRAGMENT_OFFSET = 10;
    public static final int FRAGMENT_TIME_OF_CREATION = 11;
    
    /***********************************************************/
    
    public static final int DOMAIN = 0;
    public static final int SOLUTIONS = 1;
    public static final int TIMEOUT_SEARCH = 2;
    public static final int TIMEOUT_TOTAL = 3;
    public static final int JM = 4;
    public static final int USS = 5;
    public static final int DGEQ = 6;
    public static final int DLEQ = 7;
    public static final int UNIFORM = 8;
    public static final int ELLIPSOID = 9;
    
    /***********************************************************/
    
    public static final int MONTECARLO_SAMPLING = 0;
    public static final int GIBBS_SAMPLING = 1;
    public static final int FASTA_OPTION = 2;
    public static final int RMSD_OPTION = 3;
    public static final int VERBOSE_OPTION = 4;
    public static final int GIBBS_OPTION = 5;
    public static final int CGC_OPTION = 6;
    public static final int PDB_FILE = 7;
    
    /***********************************************************/
    
    public static final String SOLVER_PATH = System.getProperty("user.dir") 
                + Utilities.getSeparator()
                + "Solver";
    
    public static final String TEMP = SOLVER_PATH
                + Utilities.getSeparator()
                + "temp"
                + Utilities.getSeparator();
    
    public static final String FIASCO_PATH = SOLVER_PATH
                + Utilities.getSeparator()
                + "Fiasco"
                + Utilities.getSeparator();
    
    public static final String COCOS_PATH = SOLVER_PATH
                + Utilities.getSeparator()
                + "Cocos"
                + Utilities.getSeparator();
    
    public static final String PROTEINS_PATH = SOLVER_PATH
                + Utilities.getSeparator()
                + "proteins"
                + Utilities.getSeparator();
    
    /***********************************************************/
    
    public static final String tab1 = "    ";
    public static final String tab2 = "       	              ";
    
    /* ViewOptionsPanel */
    public static final String NORMAL = "Normal";
    public static final String SPACEFILL = "SpaceFill";
    public static final String DOTS = "Dots";
    public static final String DOTSWIREFRAME = "DotsWireframe";
    public static final String BACKBONE = "Backbone";
    public static final String TRACE = "Trace";
    public static final String RIBBON = "Ribbon";
    public static final String CARTOON = "Cartoon";
    public static final String STRANDS = "StrandS";
    public static final String BALLANDSTICK = "BallAndStick";
    public static final String MESHRIBBONS = "MeshRibbons";
            
    /* Jmol panel */
    public static final String MOLMOUSESETUP = "unbind; bind \"RIGHT\" \"move "
            + "0 0 0 0  _DELTAX _DELTAY 0 0 0 50;\"; ";
    
    public static final String COLOR_SELECT_AN_ATOM = "color chartreuse; ";
    public static final String COLOR_ADD_FRAGMENT = "color atoms green; ";
    public static final String COLOR_COMMON_SUBSEQUENCE = "color purple; ";
    public static final String COLOR_DESELECT = "color structure; ";
    public static final String COLOR_DESELECT_FRAGMENT = COLOR_DESELECT 
            + "select *.CA; color red; ";
    
    public static final String COMMAND_PREPARE_STRUCURE_BASE = "spacefill off; "
            + "wireframe off; select backbone; wireframe 50; color structure; "
            + "select *.CA; color red; ";
    public static final String COMMAND_PREPARE_STRUCURE = "select *; "
            + "spacefill off; "
            + "wireframe off; select backbone; wireframe 50; color structure; "
            + "select *.CA; color red; select *; background HOVER red; "
            + "background labels black; select *; center;";
    public static final String COMMAND_SAVE_ON_PDB = "write pdb \"" 
            + DUMMY_PDB_FILE + "\"; ";
    //public static final String COMMAND_APPEND_ON_IN_PDB = "append pdb \"" 
    //        + IN_PDB_FILE + "\"; ";
    public static final String COMMAND_LOAD_FROM_PDB = "load pdb::" 
            + DUMMY_PDB_FILE + "; ";
    public static final String COMMAND_LOAD_APPEND_FROM_PDB = "load append pdb::" 
            + DUMMY_PDB_FILE + "; ";
    
}//Defs
