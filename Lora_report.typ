#set math.equation(numbering: "1.")

#set page(width: 210mm, height: 297mm, margin: 25mm)
#set heading(numbering: "1.")
#import "@preview/pintorita:0.1.4"
#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

= Fine-Tuning Pretrained Large Language Models 
= Table of Contents
#outline()

#outline(
  title: [List of Figures],
  target: figure.where(kind: image)
)


#pagebreak()



== Training LLMs

Training for an LLM is a long and compute intensive task. 
The level of current models such as GPT5 with 1.8trillion. This initial training
create a generalist model that struggles with specific information and
queries. A more specifically trained model is needed to allow for
accurate domain specific responses. Recreating the model for each task
in this case is highly inefficient and would lead to bloated systems
with many AIs. As every task would require its own fully trained model.

= Adapter Architecture

This deficiency in base models opened the door for the adapter
architecture to take place. This approach applies an additional layer to
the AI adapter. This adapter is a created by freezing the current model
weights, then training an additional set of weights to act upon the base
model that allows the LLM to have its weights altered only by the
adapter letting it be a plug and play solution, The matematical
representation of this is at a high level:

#figure(image("images/Base_model_fine_tuning.png", width: 40%), caption: [Fine tuning diagram])

$ \min L(D; W.x + Δ W.x) $ <fine_tune_explained>


-   L : This represents the loss function used to calculate the
    gradient that needs to be minimized for the LLM.

-   D : This represents the dataset the loss function is being
    trained on and what is being optimized for.

-   Θ : θ₀ represent the weights for base model and Δθ represent the
    weights from the fine tuning. This is how the adapter is swappable
    as the weights are not integrated into the base model.

#figure(
  image("images/2025-10-22-12:39:26.png"),
  caption : [Adapter Architecture]
)


= Fine Tuning approaches

== Full Parameter Fine tuning

Full parameter fine tuning approach was first proposed in 2018 @howard_universal_2018 called ULMFiT. This principle has been taken and applied in many forms to models such as DistilBERT@sanh_distilbert_2020 and BERT @devlin_bert_2019.
The base approach is is freezing the original weights then creating a blank matrix of the model weights, then training those to scale each weight individually to bias towards the new target. 
While efficent it is still a computationally heavy process as every single weight is modified but as a baseline it allows for the adapter architecture to be used.

The formula used to represent this would be

$ \min L(D; W.x + Δ W.x) $


The size of matrix θ₀ and Δθ are both m \* n where m and n represent the rows and columns in θ₀. The rest of the definitions are here @fine_tune_explained.
This method of training allows a small dataset to impact the results of a larger model removing the need to train the model on a huge corpus of data to get tangible results. This being the first step that allowed for fine tuning to be brought forward into conversation for all models.

#figure(
  image("images/QLoRA2.png")
  , caption: [Full, LoRA, QLoRA, Fine-Tuning Comparision @noauthor_parameter-efficient_nodate]
)

#pagebreak()
== LoRA Fine Tuning

While traditional fine-tuning updates all parameters of a pre-trained model, 
LoRA (Low-Rank Adaptation), introduced in 2021 (Hu, Shen, Wallis, Allen-Zhu, Li, Wang & Chen), 
takes a more efficient approach by freezing the original model weights and 
introducing a small number of additional trainable parameters. This design 
drastically reduces the computational and memory requirements of model adaptation.

Instead of using a weight update of $d^2$ like in Fine Tuning, LoRA modifies 
this process by decomposing the weight update $\ΔW$ into the product of two 
much smaller low-rank matrices, $\A$ and $\B$, defined as:

$\W' = W + B A$

where 

$A$ is a matrix of $n$ rows multiplied by $r$ columns ($A = n \* r$)

$B$ is a matrix of $r$ rows multiplied by $m$ columns ($B = r \* m$) 

$r$ is much smaller than $d$ ($r \<< d$)

There's visualisation of there here: @noauthor_parameter-efficient_nodate

Here, the pre-trained weights $W$ are frozen & they remain fixed during 
training and only $A$ and $B$ are updated. This means that instead of 
learning $d^2$ parameters, LoRA learns only $2\dr$, significantly reducing 
the number of trainable parameters when $r$ is small. The product $\BA$ serves 
as a low-rank approximation of $\ΔW$, capturing the essential adjustments 
needed to specialize the model for a new task without altering the base 
model directly.


== Vera Fine Tuning
Vera @kopiczko_vera_2024 fine-tuning is an innovation built on top of LoRA, designed to decrease the memory overhead associated with parameter-efficient fine-tuning.
The core of the method is based on Random Matrix Adaptation and LoRa. 


Instead of learning two low rank matrices VeRa, begins by generating two low rank matrices of sizes m \* r and r \* n, which are frozen after the initial generation.

Next two diagonal matrices are created of size m \* m and r \* r. These diagonal matrices scale the two low rank matrices. The method being similar to a switchboard where each value can amplify or deactivate sections in the low rank matrices without the need to store full parameter sets.

In mathematical terms @kopiczko_vera_2024

$ W.x + Δ W.x = W.x + Λ_b B Λ_d A x $ 

- A and B:  Are randomly generated low rank matrixes of sizes m \* r and r \* n which multiply to create the W.x matrix.

- Λ_b and  Λ_d: Are diagonal matrixes which are used to scale the A and B matrices. They are of sizes m \* m and r \* r.

Unlike traditional Lora, Vera only learns the scaling diagonal matrix values. This severely reduces the number of required parameters going from *r(m + n)* to only *m + r*.
This significant decrease in learnable parameters does come at a slight decrease of accuracy but the sheer amount of trainable parameters decreased merits this method as a clear innovation on LoRa.


#pagebreak()
== QLoRA Fine Tuning
Building upon LoRA's efficiency, QLoRA (Quantised Low-Rank Adaptation), 
introduced in 2023 (Dettmers, Pagnoni, Holtzman & Zettlemoyer), further 
optimises fine-tuning by combining LoRA with Quantised model weights. 
Where LoRA freezes the original full-precision weights and trains only 
small low-rank matrices, QLoRA first Quantises those frozen base weights 
to a 4-bit representation, drastically reducing memory usage while 
maintaining model performance. 

QLoRA uses a 4-bit data type called NormalFloat (NF4), which is 
optimised for normally distributed weights. It applies block-wise 
quantisation, normalising each block and mapping it to one of 16 
NF4 levels. The weights are stored in this compact 4-bit form and 
dequantised back to 16-bit only during computation, greatly reducing 
memory use without sacrificing performance.

In QLoRA, the pre-trained weight matrix $W$ is stored in a quantised 
form $\W₄b\it$, the fine-tuning process still operates the same on 
low-rank adapters $A$ and $B$, as in LoRA:

$\W' = \W₄b\it \+ B\*A$

where 

$A$ is a matrix of $n$ rows multiplied by $r$ columns ($A = n \* r$)

$B$ is a matrix of $r$ rows multiplied by $m$ columns ($B = r \* m$) 

$r$ is much smaller than $d$ ($r \<< d$)

There's visualisation of there here: @noauthor_parameter-efficient_nodate


The quantised weights $\W₄b\it$ remain frozen, and only the adapter 
matrices $A$ and $B$ are updated through backpropagation. 
Because quantisation compresses $W$ into a 4-bit format and LoRA 
limits the trainable parameters to $2\dr$, QLoRA achieves extreme 
memory efficiency
These improvements allow QLoRA to maintain full 16-bit fine-tuning 
performance while using a fraction of the memory and compute resources. 


== Lora vs Full fine tuning
Full fine-tuning updated all 66,955,779 parameters (100% of the model) and took 372.3 seconds of training time, achieving a validation accuracy of 84.2%. In contrast, LoRA updated only 1,181,955 parameters (1.73% of the model), using a rank of 64 for the adapter matrices, which drastically reduces the number of trainable parameters. 
#figure(
  image("images/Trainable Parameter.png", width: 90%)
  , caption: [Trainable Parameters]
)
While the training time for LoRA is only slightly faster (299.3 seconds), this is because the forward pass still computes over the full set of model parameters, but the backpropagation is much faster since gradients are only computed for a small set of LoRA parameters (trainable parameters). LoRA achieves a validation accuracy of 79.3% even with a large reduction in trainable parameters, only 4.9% below full fine-tuning. Overall, this comparison demonstrates that LoRA provides an improvement in parameter and memory efficiency, making it a better option for fine-tuning large models.
#figure(
  image("images/Validation Accuracy.png", width: 90%)
  , caption: [Trainable Parameters]
)


== Lora Innovation Vera
Lora @hu_lora_2021 is a new technology, but it has ushered in a golden age of fine-tuning models.
A strong contender to replace LoRa is VeRa @kopiczko_vera_2024.

For our comparison, the following hyperparameters were used, and the dataset \@referencehere was used:

#table(
  columns: (auto, auto, auto, auto, auto, auto),
align: horizon,
  table.header(
    [*Model*],
    [*lr*],
    [*rank*],
    [*epochs*],
    [*batch size*],
    [*maxlen*]
  ),
    [LoRA],
    [5e-4],
    [64],
    [6],
    [16],
    [256],
    [VeRa],
    [1e-3],
    [64],
    [6],
    [16],
    [256]
)

These are not optimal hyper parameters for these models but they allow for a comparison to be performed.

The key comparison is between trainable parameters. As seen in \@referencehere there is a huge disparity between regular fine tuning and LoRa.
There is a similar decrease in trainable parameters between VeRa and LoRa as well. 

Lora has 1.1 million parameters, and VeRa has 12 thousand, there is a 900x difference in learnable parameter count.
This significant decrease is monumental as VeRa only needs 0.02% of the trainable parameters to train the entire model.

This decrease does come with a slight reduction in accuracy, going from 79% to 72% in our findings \@referencehere.
This is something that while important, is not an issue as with larger models the decrease in parameters will allow for significantly more epochs with the same compute allowing VeRa to outperform LoRa.

Overall, this comparison demonstrates that while LoRa @hu_lora_2021 is still a relatively new technology, the derivative methods such as VeRa @kopiczko_vera_2024 continue to push the boundaries of parameter-efficient fine-tuning, enabling high performance with dramatically reduced memory and compute requirements.



== Conclusion



#bibliography("references.bib")
