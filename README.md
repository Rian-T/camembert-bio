<a href=https://camembert-bio-model.fr/>
  <img width="300px" src="https://silver-crostata-bec6c0.netlify.app/authors/camembert-bio/avatar_hu793b92579abd63a955d3004af578ed96_116953_270x270_fill_lanczos_center_3.png">
</a>

# CamemBERT-bio : a Tasty French Language Model Better for your Health

CamemBERT-bio is a state-of-the-art french biomedical language model built using continual-pretraining from [camembert-base](https://huggingface.co/camembert-base). 
It was trained on a french public biomedical corpus of 413M words containing scientific documments, drug leaflets and clinical cases extrated from theses and articles.
It shows 2.54 points of F1 score improvement on average on 5 different biomedical named entity recognition tasks compared to [camembert-base](https://huggingface.co/camembert-base).

## Absract

Clinical data in hospitals are increasingly accessible for research through clinical data warehouses, however these documents are unstructured. It is therefore necessary to extract information from medical reports to conduct clinical studies. Transfer learning with BERT-like models such as CamemBERT
has allowed major advances, especially for named entity recognition. However, these models are
trained for plain language and are less efficient on biomedical data. This is why we propose a new
french public biomedical dataset on which we have continued the pre-training of CamemBERT. Thus,
we introduce a first version of CamemBERT-bio, a specialized public model for the french biomedical
domain that shows 2.54 points of F1 score improvement on average on different biomedical named
entity recognition tasks.

- **pre-print:** https://hal.science/hal-04085419
- **Developed by:** [Rian Touchent](https://rian-t.github.io), [Eric Villemonte de La Clergerie](http://pauillac.inria.fr/~clerger/)
- **Logo by:** [Alix Chagué](https://alix-tz.github.io)
- **License:** MIT

## Usage

Model available at: https://hf.co/almanach/camembert-bio-base  

*evaluations scripts coming soon*
