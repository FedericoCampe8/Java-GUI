package jafatt;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JButton;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Container;
import java.awt.Toolkit;
import java.awt.Dimension;
import javax.swing.JPanel;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ConstraintInfoPanel extends JFrame implements Runnable {
    
    private MessageArea msArea;
    private JScrollPane scroll;
    private JPanel inPanel;
    private JPanel exitPanel;
    private JButton closeButton;
    
    public ConstraintInfoPanel (){
        super("Info");
        setUp();
    }
    
    private void setUp(){
                
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 50.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 75.0) / 100.0;
        
        msArea = new MessageArea();
        
        /* Add the scroll bar and set auto-scroll */
        scroll = new JScrollPane(msArea);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener(){
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e){
               //msArea.select(msArea.getHeight() + 1000, 0);
            }
        });
        
        closeButton = new JButton("Close");
        String os = System.getProperty("os.name").toLowerCase();
        if((os.indexOf("win") >= 0) || (os.indexOf("nix") >= 0))
            closeButton.setBorder(
                    new javax.swing.border.SoftBevelBorder(
                            javax.swing.border.BevelBorder.RAISED)
                    );
        closeButton.setBackground(Defs.INNERCOLOR);
        //closeButton.setToolTipText("Click to set the fragment");
        closeButton.addActionListener(new ActionListener(){
            @Override
            public void actionPerformed(ActionEvent evt) {
                exit();
            }
        });
        closeButton.setEnabled(true);
        
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setResizable(true);
        inPanel = new JPanel();
        inPanel.setLayout(new BorderLayout());
        
        exitPanel = new JPanel();
        exitPanel.setLayout(new FlowLayout());
        
        exitPanel.add(closeButton);
        
        writeInfo();
        
        inPanel.add(scroll, BorderLayout.CENTER);
        inPanel.add(exitPanel, BorderLayout.SOUTH);
        
    }
    private void exit(){
        this.dispose();
    }
    
    private void writeInfo(){
        msArea.writeln(info, false);
    }
    
    @Override
    public void run() {
	pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run
    
    private String info = " "
            + "1.  Domain Size \n\n"
            + "  Given an integer n, the domain size constraint sets the domain's \n"
            + "  size of the model to n \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --domain-size n \n"
            + "  where: \n"
            + "  - n is the size of the domain\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 2.  Max Solution \n\n"
            + "  Given an integer n the max solution constraint sets the maximum \n"
            + "  value of the solver's search space \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --ensembles n \n"
            + "  where: \n"
            + "  - n is the maximum number of solutions explored by the solver\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 3.  Timeout Search \n\n"
            + "  Given an integer n, the timeout search constraint bounds the time \n"
            + "  the solver uses for exploring the search space \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --timeout-search n \n"
            + "  where: \n"
            + "  - n is the time bound (in seconds)\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 4.  Timeout Total \n\n"
            + "  Given an integer n, the timeout total constraint bounds the overall \n"
            + "  time of the process (i.e. search time, jmf propagation, etc.) \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --timeout-total n \n"
            + "  where: \n"
            + "  - n is the time bound (in seconds)\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 5. Distance Greater than equal \n\n"
            + "  Given a distance n and two amino acids a_i a_j, the distance less \n"
            + "  than equal constraint sets the distance between two fixed amino \n"
            + "  acids to be greater than n\n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --distance-geq a_i a_j n \n"
            + "  where: \n"
            + "  - a_i, a_j are two amino acids \n"
            + "  - n is the distance forced between the amino acids\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 6. Distance Less than equal \n\n"
            + "  Given a distance n and two amino acids a_i a_j, the distance less \n"
            + "  than equal constraint sets the distance between two fixed amino \n"
            + "  acids to be less than n \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --distance-leq a_i a_j n \n"
            + "  where \n"
            + "  - a_i, a_j are two amino acids \n"
            + "  - n is the distance forced between the amino acids\n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 7. Unique Constraint \n\n"
            + "  The unique constraint is a spatial constraint which guarantees that a given \n" 
            + "  discretization for a chain of adjacent points is visited in at most one \n"
            + "  solution. \n\n"
            + "  SEMANTICS: \n\n"
            + "  Consider a 3D lattice approximating the 3D space. We say that gamma(aa_i) \n"
            + "  gives the lattice position for the amino acid aa_i. An assignment (x_1, ..., x_n) \n"
            + "  of variables P_1, .., P_n satisfy the  unique constraint over a sequence \n" 
            + "  of amino acids aa_1, .., aa_n,  if there is no other solution SOL_k \n"
            + "  s.t. (gamma_k(aa_1) = x_1 [and] ... [and] gamma_k(aa_n) = x_n. \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --uniform aa_1 .. aa_n : voxel-side= K [center= X Y Z ] \n"
            + "  where: \n"
            + "  - aa_1 .. aa_n, is a list of amino acids (starting from 0) \n"
            + "    for which CAs will be involved in the uniform constraint (every a_i will be \n"
            + "    placed using one grid) \n"
            + "  - K, is the side of a voxel in Angstroms. \n\n"
            + " NOTE: \n\n"
            + " The lattice side is now fixed to 200 Angstroms. If you experience \n"
            + " any problem with the lattice size, please enlarge the lattice side. \n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 8. Ellipsoid \n\n"
            + "  Given a list of amino acid A_1, ..., A_k, (k>=1) two focus f1, f2, and the sum \n"
            + "  of the radii it specify an area in which the amino acid 'A_i' should be bound in. \n\n"
            + "  INPUT SYNTAX:\n\n"
            + "  --ellipsoid aa_1 .. aa_n : f1= X Y Z f2= X Y Z sum-radii= K \n"
            + "  where: \n"
            + "  - aa_1 .. aa_n, is a list of amino acids (starting from 0) \n "
            + "    for which CAs will be involved in the constraint \n"
            + "  - f1 and f2 X Y Z, are the coordinates for the two focus \n"
            + "  - K, is the radius sum \n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 9. JM constraint \n\n"
            + "  The Joined Multibody constraint, is set between two anchor points: front- and \n"
            + "  end-anchor (included) to generate non-redundant paths between them. It uses an \n"
            + "  approximated propagation algorithm (JMf) that heavily relies on some \n"
            + "  clustering procedure. \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --jm aa_s {-> | <- | <->} aa_f : numof-clusters= K_min K_max \n"
            + "                                   sim-params= R B \n"
            + "                                   tollerances= eps_R eps_B \n"
            + "  where: \n"
            + "  - aa_s: is the amino acid (starting from 0) corresponding to the front-anchor \n"
            + "    of the  flexible chain. \n"
            + "  - aa_f: is the amino acid relative to the end-anchor of the flexible \n"
            + "    chain. \n"
            + "  - '->' | '<-' | '<->', defines the propagation direction, either MONO- or \n"
            + "    BI-directional \n"
            + "  - K_min, K_max: are the minimum / maximum number of cluster to generate \n"
            + "  - R, B: are the sim-clustering parameters eps_R, eps_B: are the maximum \n"
            + "    tolerances radius (eps_R) and orientation (eps_B) within which an \n"
            + "    end-anchor can be placed. \n\n"
            + "  DEFAULT VALUES: \n\n"
            + "  - K_min = max {50, dom-size} \n"
            + "  - K_max = 500 \n"
            + "  - R = 0.5 A \n"
            + "  - B = 15 deg \n"
            + "  - eps_R = 1.5 A \n"
            + "  - eps_R = 60 deg \n\n"
            + "  EXAMPLE: \n\n"
            + "  --jm 88 '->' 96 : numof-clusters= 60 200 \n"
            + "                    sim-param= 0.5 15 \n"
            + "                    tolerances= 1.5 30 \n"
            + " \n-----------------------------------------------------------------"
            + "----------------------------------------------------------------- \n\n"
            + " 10. Unique Source-Sink constraint \n\n"
            + "  The unique-source-sink constraint between the varaibles V_i and V_j is a \n"
            + "  cardinaliry constraints which given an assignment for V_i, ensures that in the \n"
            + "  solutions pool there is at most one assignment for V_j, for each grid voxel of \n"
            + "  G_j. This constraint associates a grid to the variable V_j. \n\n"
            + "  SEMANTICS: \n\n"
            + "  Fixed V_i, [not][exists] V_j^m, V_j^n s.t. find_voxel(V_j^m) = find_voxel(V_j^n) \n"
            + "  where: \n"
            + "  - V_j^{k} represents the assignment for variable V_j in the k-th solution \n"
            + "    generated. \n"
            + "  - find_voxel : V -> N is a function which associate each variable \n"
            + "    assignemnt with its grid voxel index (if any). \n\n"
            + "  INPUT SYNTAX: \n\n"
            + "  --unique-source-sinks a_i '->' a_j : voxel-side= K \n"
            + "  where: \n"
            + "  - a_i, is the source atom (the CA_i, with i >= 0)  \n"
            + "  - a_j, is the sink atom (the CA_j, with i >= 0) \n"
            + "  - K, is the side of a voxel in Angstroms. \n\n"
            + "  NOTE: \n\n"
            + "  The lattice side is now fixed to 200 Angstroms. If you experience any problem \n"
            + "  with the lattice size, please enlarge the lattice side. \n"
            + "   To exploit the synergy of the JM and the unique_source_sink constraints use the \n"
            + "  following parameters: \n"
            + "  if JM is set on: AA_i -> AA_j \n"
            + "  set unique_source_sink on: AA_{i-1} -> AA_j \n";
    
}
