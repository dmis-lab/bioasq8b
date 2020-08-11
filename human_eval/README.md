# Human Evaluation
We evaluated the statistics of unanswerable data in the extractive QA setting. The table below shows the unanswerable rate in all batches of 7B test datasets only for factoid and list type questions. We calculated the rate as a criterion of _Ground Truth Answer cannot be exactly match in Human Annotated Corpus (Snippet)_.

Here are the result table from the BioASQ Challenge Task 7B (Phase B) test dataset.
Type | 7B Batch1 | 7B Batch2 | 7B Batch3 | 7B Batch4 | 7B Batch5 | 7B Total
---|:---:|:---:|:---:|:---:|:---:|:---:
Factoid | 0.359 (14/39) | 0.120 (3/25) | 0.310 (9/29) | 0.118 (4/34) | 0.229 (8/35) | 0.216 (35/162)
List | 0.083 (1/12) | 0.235 (4/17) | 0.200 (5/25) | 0.136 (3/22) | 0.500 (6/12) | 0.204(18/88)


And several unanswerable cases : (<b>Bold</b> and <ins>underline</ins> denotes no exact match and exact match in lowercase respectively.)

<details markdown="1">
<summary>Factoid</summary>
<pre>
  <code>
    ID : 5c5f094c1a4c55d80b00000c
    Question : Where is the protein Bouncer located?
    Ground Truth Answer : <b>Bouncer is membrane bound.</b>
    Context : <b>Membrane-bound Bouncer</b> 
    <br />
    <br />
    ID : 5c5f264b1a4c55d80b00001e
    Question : What is the function of Plasminogen activator inhibitor 1?
    Ground Truth Answer : <b>PAI-1 is important in fibrinolysis.</b>
    Context : Plasminogen activator inhibitor-1 (<b>PAI-1</b>) <b>is</b> an <b>important</b> physiological inhibitor of tissue-type plasminogen activator (tPA) and plays a critical role <b>in fibrinolysis.</b>
    <br />
    <br />
    ID : 5c5219f67e3cb0e231000006
    Question : As of September 2018, what machine learning algorithm is used to for cardiac arrhythmia detection from a  short single-lead ECG recorded by a wearable device?
    Ground Truth Answer : <b>SVM OR Support Vector Machine</b>
    Context : <b>SVM</b> approach for cardiac arrhythmias detection in short single-lead ECG recorded by a wearable device
    <br />
    <br />
    ID : 5c531d8f7e3cb0e231000017
    Question : What does the pembrolizumab companion diagnostic test assess?
    Ground Truth Answer : <b>transglutaminase-1 gene (TGM1) mutations</b>
    Context : Bathing suit ichthyosis (BSI) is an uncommon phenotype classified as a minor variant of autosomal recessive congenital ichthyosis (ARCI). OBJECTIVES: We report a case of BSI in a 3-year-old Tunisian girl with a novel <b>mutation of the transglutaminase 1 gene (TGM1)</b>
    <br />
    <br />
    ID : 5c54d1a207647bbc4b000007
    Question : What is nyctinasty in plants?
    Ground Truth Answer : <b>movement of leguminous plants in response to darkness</b>
    Context : <b>Leguminous plants open their leaves during the daytime and close them at night as if sleeping</b>, a type of movement that follows circadian rhythms, and is known as nyctinastic <b>movement</b>
    <br />
    <br />
    ID : 5c7d5fcfd774d04240000011
    Question : Where are pannexins localized?
    Ground Truth Answer : <b>In membranes</b>
    Context : Pannexins are a family of integral <b>membrane</b> proteins with distinct post-translational modifications, sub-cellular localization and tissue distribution. 
    <br />
    <br />
    ID : 5c9a6693ecadf2e73f000031
    Question : What is a mitosome?
    Ground Truth Answer : <b>Simple and anaerobic mitochondria.</b>
    Context : Mitosomes are the <b>simplest</b> and the least well-studied type of <b>anaerobic mitochondria.</b>
    <br />
    <br />
    ID : 5c34ad63da8336e21a000007
    Question : Which genomic positions are preferentially selected for transposon insertion?
    Ground Truth Answer : <b>non coding DNA, heterochromatin</b> 
    Context : The FISH analysis of the pepper Tat elements showed a random distribution in <b>heterochromatic</b> and euchromatic regions, whereas the tomato Tat elements showed heterochromatin-preferential accumulation
    <br />
    <br />
    ID : 5c632542e842deac6700000d
    Question : Which approach was used to diagnose a patient with Cutis Verticis Gyrata-Intellectual Disability (CVG-ID) syndrome?
    Ground Truth Answer : <ins>Magnetic Resonance Imaging,</ins> <b>MRI</b> 
    Context : Here, we report a patient with CVG-ID that was diagnosed using the novel approach of <ins>magnetic resonance imaging</ins> and we conduct a systematic review of all patients reported in the last 60 years, discussing the core clinical features of this syndrome.
    <br />
    <br />
    ID : 5c7839edd774d04240000003
    Question : When is serum AFP used as marker?
    Ground Truth Answer : <ins>in HCC,</ins> <b>Hepato cellular carcinoma</b>
    Context : AFP serum were considered independent predictors for macrovascular invasion <ins>in HCC</ins> patients
    <br />
    <br />
    ID : 5c51fe8907ef653866000007
    Question : What is the percentage of individuals at risk of dominant medically actionable disease?
    Ground Truth Answer : <b>1/38,</b> <ins>2.7%</ins>
    Context : Our study shows that 1 in 38 healthy individuals (<ins>2.7%</ins>) has a (likely) pathogenic variant in one of 59 medically actionable dominant disease genes for which the American College of Medical Genetics and Genomics (ACMG) recommends disclosure.
  </code>
</pre>
</details>

<details markdown="2">
<summary>List</summary>
<pre>
  <code>
    ID : 5c641179e842deac67000012
    Question : What are the effects of STEF depletion?
    Ground Truth Answer : <b>Reduction of apical perinuclear actin cables, Decrease of nuclear stiffness, Reduction of expression of TAZ-regulated genes</b>
    Context : The mechanisms regulating the actin cap are currently poorly understood. Here, we demonstrate that STEF/TIAM2, a Rac1 selective guanine nucleotide exchange factor, localises at the nuclear envelope, co-localising with the key perinuclear proteins Nesprin-2G and Non-muscle myosin IIB (NMMIIB), where it regulates perinuclear Rac1 activity. We show that STEF depletion <b>reduces apical perinuclear actin cables</b> (a phenotype rescued by targeting active Rac1 to the nuclear envelope), increases nuclear height and impairs nuclear re-orientation. STEF down-regulation also reduces perinuclear pMLC and decreases myosin-generated tension at the nuclear envelope, suggesting that STEF-mediated Rac1 activity regulates NMMIIB activity to promote stabilisation of the perinuclear actin cap. Finally, STEF depletion <b>decreases nuclear stiffness</b> and <b>reduces expression of TAZ-regulated genes</b>, indicating an alteration in mechanosensing pathways as a consequence of disruption of the actin cap.
    <br />
    <br />
    ID : 5c5214207e3cb0e231000003
    Question : List potential reasons regarding why potentially important genes are ignored
    Ground Truth Answer : <b>Identifiable chemical properties, Identifiable physical properties, Identifiable biological properties</b>, <ins>Knowledge about homologous genes from model organisms</ins>
    Context : Here, we demonstrate that these differences in attention can be explained, to a large extent, exclusively from a small set of <b>identifiable chemical, physical, and biological properties</b> of genes. Together with <ins>knowledge about homologous genes from model organisms</ins>, these features allow us to accurately predict the number of publications on individual human genes, the year of their first report, the levels of funding awarded by the National Institutes of Health (NIH), and the development of drugs against disease-associated genes.
    <br />
    <br />
    ID : 5c72b6be7c78d69471000072
    Question : Which enzymes are inhibited by Duvelisib?
    Ground Truth Answer : <b>phosphoinositide 3-kinase-\u03b3</b>, <ins>phosphoinositide 3-kinase-\u03b4</ins>
    Context : Duvelisib is an oral dual inhibitor of <ins>phosphoinositide 3-kinase-\u03b4</ins> (PI3K-\u03b4) and <b>PI3K-\u03b3</b> in late-stage clinical development for hematologic malignancy treatment.
    <br />
    <br />
    ID : 5c9fb583ecadf2e73f000042
    Question : What are 5 key questions in human performance modeling?
    Ground Truth Answer : <b>Why build models?, What are the expectations of a good model?, What are the procedures and requirements?, How do we integrate a model with system design?, What are the future directions of Human performance modeling?</b>
    Context : the five key questions of human performance modeling: 1) <b>Why</b> we <b>build models</b> of human performance; 2) <b>What the expectations of a good</b> human performance <b>model are</b>; 3) <b>What the procedures and requirements</b> in building and verifying a human performance model <b>are</b>; 4) <b>How we integrate a</b> human performance <b>model with system design</b>; and 5) <b>What the possible future directions of human performance modeling</b> research <b>are</b>.
  </code>
</pre>
</details>
