# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: The positive class is a patient being readmitted to the hospital within 30 days, missed positives are more concerning because not identifying conditions on a patient could prevent them from receiving the proper follow up and care once discharged. Although false alarms are undesired, with extra costs and unnecessary treatments, missed positives can lead to more serious outcomes.
- diabetes: The positive class is a patient that was screened positive for diabetes. Missed positives matter more because failing to identify diabetes could delay proper care and treatment which can lead to serious complications. False alarms can lead to extra testing, but is overall less harmful.

## Recommendations
- readmission: None is adequate for definite use without further validation. The default 0.5 threshold is inadequate because missing high-risk patients is of major concern. High accuracy can be misleading when readmissions are uncommon, therefore a model should show better sensitivity and keep false alarms low.
- diabetes: None is adequate for definite use without further validation. However, a model with a threshold that achieves at least 0.90 sensitivity with minimal false positives would be prioritized. Further validation is necessary before a recommendation is made.

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: I chose to display the confusion matrix and key performance metrics side by side in the 'Beyond Accuracy' section because it allows for easier comparison without having to scroll between different sections. 
- AI suggestion I turned down: Codex suggested adding interactive ROC curves to compare the classification model. I decided to turn this suggestion down because it was an uncessary feature as the existing AUROC comparison table already provides the necessary information to evaluate model discrimination.
