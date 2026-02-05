 Directory Structure                                                                                                  
                                                                                                                       
  tutorials/Ramp/steady_SA/                                                                                            
  ├── train/                                                                                                           
  │   ├── c1/                      # Training case 1 (U0=10 m/s)                                                       
  │   │   ├── 0_orig/              # Initial fields (U, p, nuTilda, nut, k, omega, betaFINuTildaData)                  
  │   │   ├── constant/            # turbulenceProperties (SA, SST variants), transportProperties                      
  │   │   ├── system/              # controlDict, fvSchemes, fvSolution, decomposeParDict                              
  │   │   └── runPrimal.py         # Primal solver script                                                              
  │   ├── c2/                      # Training case 2 (U0=20 m/s)                                                       
  │   │   └── (same structure as c1)                                                                                   
  │   ├── tf_training/                                                                                                 
  │   │   └── trainModel.py        # TensorFlow ML training for decoupled FIML                                         
  │   ├── runScript.py             # Coupled FIML optimization                                                         
  │   ├── runScript_FI.py          # Decoupled Field Inversion                                                         
  │   ├── probePointCoords.json    # Probe point locations                                                             
  │   ├── preProcessing.sh         # Generate SST reference data                                                       
  │   └── Allclean.sh                                                                                                  
  │                                                                                                                    
  └── predict/                                                                                                         
      ├── baseline/                # SA without augmentation (U0=15 m/s)                                               
      ├── trained/                 # SA with trained NN                                                                
      ├── reference/               # SST reference                                                                     
      ├── Allrun.sh                                                                                                    
      └── Allclean.sh                                                                                                  
                                                                                                                       
  Key Differences from k-omega Tutorial                                                                                
  ┌───────────────────────┬───────────────────────┬────────────────────────┐                                           
  │        Aspect         │   k-omega (steady)    │     SA (steady_SA)     │                                           
  ├───────────────────────┼───────────────────────┼────────────────────────┤                                           
  │ Turbulence model      │ kOmega                │ SpalartAllmaras        │                                           
  ├───────────────────────┼───────────────────────┼────────────────────────┤                                           
  │ Augmentation variable │ betaFIK, betaFIOmega  │ betaFINuTilda          │                                           
  ├───────────────────────┼───────────────────────┼────────────────────────┤                                           
  │ Features              │ PoD, VoS, PSoSS, KoU2 │ PoD, VoS, chiSA, PSoSS │                                           
  ├───────────────────────┼───────────────────────┼────────────────────────┤                                           
  │ Number of NN models   │ 2 (k and omega)       │ 1 (nuTilda)            │                                           
  ├───────────────────────┼───────────────────────┼────────────────────────┤                                           
  │ Reference model       │ kOmegaSST             │ kOmegaSST              │                                           
  └───────────────────────┴───────────────────────┴────────────────────────┘                                           
  Workflow                                                                                                             
                                                                                                                       
  Coupled FIML:                                                                                                        
  cd train                                                                                                             
  ./preProcessing.sh              # Generate SST reference data                                                        
  mpirun -np 4 python runScript.py  # Train NN with adjoint gradients                                                  
                                                                                                                       
  Decoupled FIML:                                                                                                      
  cd train                                                                                                             
  ./preProcessing.sh              # Generate SST reference data                                                        
  mpirun -np 4 python runScript_FI.py -index 0  # Field inversion for c1                                               
  mpirun -np 4 python runScript_FI.py -index 1  # Field inversion for c2                                               
  cd tf_training && python trainModel.py        # Train NN offline                                                     
                                                                                                                       
  Prediction:                                                                                                          
  cd predict                                                                                                           
  ./Allrun.sh  # Runs baseline, reference, and trained cases