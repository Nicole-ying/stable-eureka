# EG-RSA-V2 Paper Rewrite

原始论文目录已备份到：

```text
/home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper_backup_before_eg_rsa_v2_20260611_203357
```

新生成文件：

```text
/home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper/main_eg_rsa_v2.tex
/home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper/eg_rsa_v2_refs.bib
/home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper/sections_eg_rsa_v2
```

编译命令：

```bash
cd /home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper
latexmk -pdf main_eg_rsa_v2.tex
```

如果没有 latexmk：

```bash
cd /home/utseus22/stable-eureka-nicole/paper_rewriting_output/final_paper
pdflatex main_eg_rsa_v2.tex
bibtex main_eg_rsa_v2
pdflatex main_eg_rsa_v2.tex
pdflatex main_eg_rsa_v2.tex
```
