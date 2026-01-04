# fungitastic-classification-datasci207-Fall-2025
## Team Members: 
- Noshen Habib (noshen@berkeley.edu) (git: noshen-atashe)
- Jeremy Cui (jaycray0987@berkeley.edu) (git: jcui2001)
- Daniel Motoc (daniel_motoc@berkeley.edu) (git: danmot-98) 
## Resources: 
- Main research paper: FungiTastic: [A Multi-Modal Dataset and Benchmark for Image Categorization.](https://arxiv.org/pdf/2408.13632)
- Related git repo: https://github.com/BohemianVRA/FungiTastic/tree/main 


# Run in virtual RunPod instructions

After connecting to runtime shell:
1. create virtual env $ python -m venv .venv
2. clone repo into virtual workspace
3. install requirements_remote_final.txt $ pip install -r requirements_remote_final.txt
4. run training/tuning: python src/train.py --data-config <PATH TO DATA CONFIG>