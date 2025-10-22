#set math.equation(numbering: "1.")

= Fine-Tuning Pretrained Large Language Models 


== Training LLMs

Training for an LLM is a long and compute intensive task. 
The level of current models such as GPT5 with 1.8trillion. This initial training
create a generalist model that struggles with specific information and
queries. A more specifically trained model is needed to allow for
accurate domain specific responses. Recreating the model for each task
in this case is highly inefficient and would lead to bloated systems
with many AIs. As every task would require its own fully trained model.
#figure(
  image("images/2025-10-22-12:39:20.png"),
  caption : [One model for every task]
)

= Adapter Architecture

This deficiency in base models opened the door for the adapter
architecture to take place. This approach applies an additional layer to
the AI adapter. This adapter is a created by freezing the current model
weights, then training an additional set of weights to act upon the base
model that allows the LLM to have its weights altered only by the
adapter letting it be a plug and play solution, The matematical
representation of this is at a high level:

#figure(image("images/Base_model_fine_tuning.png", width: 60%), caption: [Fine tuning diagram])

$ \min L(D; W.x + Δ W.x) $ <fine_tune_explained>


-   L : This represents the loss function used to calculate the
    gradient that needs to be minimized for the LLM.

-   D : This represents the dataset the loss function is being
    trained on and what is being optimized for.

-   Θ : θ₀ represent the weights for base model and Δθ represent the
    weights from the fine tuning. This is how the adapter is swappable
    as the weights are not integrated into the base model.

#figure(
  image("images/2025-10-22-12:39:26.png", width:60%)
  , caption: [Adapter pattern]
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



=======
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

#figure(
  image("images/LoRA.png")
  , caption: [LoRA Lower Dimensionality]
)

Here, the pre-trained weights $W$ are frozen & they remain fixed during 
training and only $A$ and $B$ are updated. This means that instead of 
learning $d^2$ parameters, LoRA learns only $2\dr$, significantly reducing 
the number of trainable parameters when $r$ is small. The product $\BA$ serves 
as a low-rank approximation of $\ΔW$, capturing the essential adjustments 
needed to specialize the model for a new task without altering the base 
model directly.


#pagebreak()
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

#figure(
  image("images/QLoRA3.png")
  , caption: [Full, LoRA, QLoRA, Fine-Tuning Comparision]
)


The quantised weights $\W₄b\it$ remain frozen, and only the adapter 
matrices $A$ and $B$ are updated through backpropagation. 
Because quantisation compresses $W$ into a 4-bit format and LoRA 
limits the trainable parameters to $2\dr$, QLoRA achieves extreme 
memory efficiency
These improvements allow QLoRA to maintain full 16-bit fine-tuning 
performance while using a fraction of the memory and compute resources. 

Because quantization compresses $W$ into a 4-bit format and LoRA 
limits the trainable parameters to $2\dr$, QLoRA achieves extreme 
memory efficiency, while still maintaining full 16-bit fine-tuning 
performance, using only a fraction of memory and compute resources.

== Lora vs Full fine tuning

== Lora Vs Vera

#bibliography("references.bib")
=======
