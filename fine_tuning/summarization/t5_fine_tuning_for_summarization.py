!pip install rouge_score
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


# 1. DATA PREPARATION
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

'''
billsum['train'][0]
{'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 5100 of the Civil Code is amended to read:\n5100.\n(a) Notwithstanding any other law or provision of the governing documents, elections regarding assessments legally requiring a vote, the election and removal of directors, amendments to the governing documents, or the grant of exclusive use of common area pursuant to Section 4600 shall be held by secret ballot in accordance with the procedures set forth in this article.\n(b) This article also governs an election on any topic that is expressly identified in the operating rules as being governed by this article.\n(c) The provisions of this article apply to both incorporated and unincorporated associations, notwithstanding any contrary provision of the governing documents.\n(d) The procedures set forth in this article shall apply to votes cast directly by the membership, but do not apply to votes cast by delegates or other elected representatives.\n(e) In the event of a conflict between this article and the provisions of the Nonprofit Mutual Benefit Corporation Law (Part 3 (commencing with Section 7110) of Division 2 of Title 1 of the Corporations Code) relating to elections, the provisions of this article shall prevail.\n(f) Directors shall not be required to be elected pursuant to this article if the governing documents provide that one member from each separate interest is a director, or if the election of directors is uncontested. For purposes of this subdivision, an election of directors is uncontested if the number of candidates for election, including write-in candidates, if applicable, does not exceed the number of directors to be elected at that election and the association has declared the election is uncontested.\n(1) An association may declare an election of directors is uncontested only if all of the following procedures have been satisfied:\n(A) The election rules required by Section 5105 have been adopted and complied with for the election.\n(B) All declared candidates were nominated before the deadline for nominations and in accordance with all lawful provisions of the association’s governing documents.\n(C) The inspector of elections has informed the board that the number of candidates does not exceed the number of directors to be elected at that election.\n(D) The board votes in open session to declare the election is uncontested after a hearing during an open board meeting where members are able to make objections to the board making that declaration.\n(E) At least 20 days before the board meeting for the vote to declare the election is uncontested, the association provides general notice to all members as set forth in Section 4045 of all of the following:\n(i) The intention of the board to vote at a regular board meeting to declare the election of directors is uncontested, and giving date, time, and place of that board meeting.\n(ii) A disclosure to members of the names of all candidates, however nominated, including self-nomination, who will be declared elected if the board declares the election is uncontested.\n(iii) The right of any member to appear at the board meeting and make an objection to the board declaring the election is uncontested before the board votes on the matter.\n(F) The names of all candidates, however nominated, the general notice required by subparagraph (E), any objection to the board making the declaration that the election of directors is uncontested, and the board vote declaring the election of directors is uncontested shall be recorded in the meeting minutes.\n(2) (A) If the association’s governing documents provide for write-in votes on the ballot, the association shall allow 15 days after the board meeting described in subparagraph (D) of paragraph (1) for a write-in candidate to submit his or her name to the inspector of elections. In the event one or more write-in candidates are timely submitted and additional candidates result in the total number of candidates exceeding the number of directors to be elected at that election, an election shall be held pursuant to general election rules as provided in this article. If after the 15-day period the total number of candidates, including the number of write-in candidates, does not exceed the number of directors to be elected at that election, the uncontested election results shall be sealed and become effective immediately, with any write-in candidates added as members. The new board shall take office immediately following the sealing of the election.\n(B) If an association’s governing documents do not provide for write-in votes on the ballot, as provided by subparagraph (A), then the association must provide at least 15 days’ general notice of a self-nomination process following the board determination described in subparagraph (D) of paragraph (1).\nSEC. 2.\nSection 5105 of the Civil Code is amended to read:\n5105.\n(a) An association shall adopt rules, in accordance with the procedures prescribed by Article 5 (commencing with Section 4340) of Chapter 3, that do all of the following:\n(1) Ensure that if any candidate or member advocating a point of view is provided access to association media, newsletters, or Internet Web sites during a campaign, for purposes that are reasonably related to that election, equal access shall be provided to all candidates and members advocating a point of view, including those not endorsed by the board, for purposes that are reasonably related to the election. The association shall not edit or redact any content from these communications, but may include a statement specifying that the candidate or member, and not the association, is responsible for that content.\n(2) Ensure access to the common area meeting space, if any exists, during a campaign, at no cost, to all candidates, including those who are not incumbents, and to all members advocating a point of view, including those not endorsed by the board, for purposes reasonably related to the election.\n(3) Specify the qualifications for candidates for the board and any other elected position, and procedures for the nomination of candidates, consistent with the governing documents. A nomination or election procedure shall not be deemed reasonable if it disallows any member from nominating himself or herself for election to the board.\n(4) Specify the qualifications for voting, the voting power of each membership, the authenticity, validity, and effect of proxies, and the voting period for elections, including the times at which polls will open and close, consistent with the governing documents.\n(5) Specify a method of selecting one or three independent third parties as inspector or inspectors of elections utilizing one of the following methods:\n(A) Appointment of the inspector or inspectors by the board.\n(B) Election of the inspector or inspectors by the members of the association.\n(C) Any other method for selecting the inspector or inspectors.\n(6) Allow the inspector or inspectors to appoint and oversee additional persons to verify signatures and to count and tabulate votes as the inspector or inspectors deem appropriate, provided that the persons are independent third parties.\n(7) Ensure that an announcement of an election and notification of nomination procedures, including self-nomination, shall be provided to all members by general notice as set forth in Section 4045 at least 60 days before any election for directors.\n(8) Ensure a member\nin good standing,\nwho satisfies\nany\nthe\nlawful\nrequirements specified\nqualifications adopted pursuant to paragraph (3) and\nby the association’s governing documents, shall not be denied the right to\nvote or the right to\nbe a candidate for director.\n(9) Ensure a member who satisfies the lawful qualifications adopted pursuant to paragraph (4) and by the association’s governing documents shall not be denied the right to vote.\n(b) Notwithstanding any other law, the rules adopted pursuant to this section may provide for the nomination of candidates from the floor of membership meetings or nomination by any other manner. Those rules may permit write-in candidates for ballots.\nSEC. 3.\nSection 5145 of the Civil Code is amended to read:\n5145.\n(a) A member of an association may bring a civil action for declaratory or equitable relief for a violation of this article by the association, including, but not limited to, injunctive relief, restitution, or a combination thereof, within one year of the date the cause of action accrues. Upon a finding that the election procedures of this article, or the adoption of and adherence to rules provided by Article 5 (commencing with Section 4340) of Chapter 3, were not followed, a court may void any results of the election.\n(b) A member who prevails in a civil action to enforce the member’s rights pursuant to this article shall be entitled to reasonable attorney’s fees and court costs, and the court may impose a civil penalty of up to five hundred dollars ($500) for each violation, except that each identical violation shall be subject to only one penalty if the violation affects each member of the association equally. A prevailing association shall not recover any costs, unless the court finds the action to be frivolous, unreasonable, or without foundation.\n(c) A cause of action under Sections 5100 to 5130, inclusive, with respect to access to association resources by a candidate or member advocating a point of view, the receipt of a ballot by a member, the counting, tabulation, or reporting of, or access to, ballots for inspection and review after tabulation, or a violation of a rule required by Section 5105 may be brought in small claims court if the amount of the demand does not exceed the jurisdiction of that court.',
 'summary': 'The Davis-Stirling Common Interest Development Act defines and regulates common interest developments that are not a commercial or industrial common interest development. The act requires a common interest development to be managed by an association, requires the association to select one or 3 independent 3rd parties as an inspector or inspectors of elections, and generally requires the association’s elections regarding assessments legally requiring a vote, the election and removal of directors, amendments to the governing documents, or the grant of exclusive use of common area, to be conducted by the inspector or inspectors of elections in accordance with specified rules and procedures. The act excepts from these election requirements an election of directors if the governing documents of the association provide that one member from each separate interest is a director.\nThis bill would additionally except from those election requirements an election of directors if the election is uncontested, as defined, and would provide a procedure for an election to be declared as uncontested. The bill adds 2 additional election requirements that would ensure an announcement of an election and notification of nomination procedures is provided in a specific manner and would ensure a member\nin good standing\nwho meets specified qualification requirements\nis not denied the right to vote or the right to be a candidate for director. The bill would authorize a cause of action alleging a violation of these and other specified election requirements to be brought in small claims court if the amount of the demand does not exceed the jurisdiction of that court.',
 'title': 'An act to amend Sections 5100, 5105, and 5145 of the Civil Code, relating to common interest developments.'
}
'''

checkpoint = "D:\work\\t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_billsum = billsum.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


# 2. DEFINE ACCURACY METRICS
rouge = evaluate.load("rouge")
# in offline mode:-
# a. download the rouge.py file manually from https://github.com/huggingface/evaluate/tree/main/metrics/rouge/rouge.py
# b. rouge = evaluate.load("local/path/to/rouge.py")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



# 3. TRAIN MODEL
# Tips and Tricks:
# a. If you get eval_loss = nan, try setting fp16=False

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="D:\work\\t5-small-billsum",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


# 4. MODEL INFERENCE
from transformers import pipeline

text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

tokenizer = AutoTokenizer.from_pretrained("D:\work\\t5-small-billsum\checkpoint-186")
model = AutoModelForSeq2SeqLM.from_pretrained("D:\work\\t5-small-billsum\checkpoint-186")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
summarizer(text)
