
# Automatic Fake News Detection: Are current models fact-checking or gut-checking? 

Ian Kelk, Wee Yi (Jack) Lee, and Benjamin Basseri 
 <br>

1. The code ran under Google colab, if you are not running from Google Colab, make sure you remove the Google Drive mounting cell.
2. Please ensure you point to the right system path directory in main.ipnyb. analyze_snes.ipynb and analyze_pomt.ipnyb. For example, in Google Colab: 
	- os.chdir('/content/drive/MyDrive/NLPProject/code/bias' )
	- sys.path.append('/content/drive/MyDrive/NLPProject/code')
	- sys.path.append('/content/drive/MyDrive/NLPProject/code/bias/')
	
3. When you run the code for the first time, it might take some time to download pretrained language models.
4. Run main.ipynb to generate all the results. When run in Google Colab Pro+, it will take about 80 - 100 hours to complete.
5. If you wish to select certain setup, you can change the below variables. For example:
	- steps = [['EMO_LEXI'], ['EMO_INT']]
	- modes = ['bert']
	- datasets = ['pomt']
	- inputtypes = ['CLAIM_AND_EVIDENCE', 'EVIDENCE_ONLY', 'CLAIM_ONLY']

6. After running main.ipynb, create the plots based on your results by running: analyze_snes.ipynb for Snopes dataset and analyze_pomt.ipnyb for PolitiFact dataset
7. To create the plots, if you don't have the full results, you will need to modify the below variables in analyze_snes.ipynb and analyze_pomt.ipnyb in order to run them. 
	- steps = ['none', 'neg', 'stop', 'pos', 'stem', 'all', 'pos-neg', 'pos-stop','formal', 'informal', 'pos-neg-stop', 'EMOINT', 'EMO_LEXI']
	- steps_dic = {'none':'none', 'neg':'neg', 'stop':'stop', 'pos':'pos', 'stem':'stem', 'all':'all', 
             			'pos-neg':'pos+neg', 'pos-stop':'pos+stop', 'formal':'formal', 'informal':'informal',
           			'pos-neg-stop': 'pos+neg+stop', 'EMOINT': 'Emotion intensity', 'EMO_LEXI': 'Emotion Lexicon'}
	- inputtypes = ["CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"]
	- inputtypes_dic = {"CLAIM_ONLY": "Claim", "EVIDENCE_ONLY": "Evidence", "CLAIM_AND_EVIDENCE": "Claim+Evidence"}

8. If you wish go generate neutralization dataset, please follow the instructions in this github: https://github.com/weeyilee/neutralization
