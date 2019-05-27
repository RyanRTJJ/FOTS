Based on original work by Ning Lu (https://github.com/jiangxiluning) and DongLiang Ma (https://vipermdl.github.io/)
as well as other open-source implementations of FOTS. Many thanks to all the collaborators and contributors. 

Example images (imgs/example/scenario_1/1.png), (imgs/example/scenario_1/2.png).

To run the training function, simply run:
python train.py -c /full/path/to/config.json

To evaluate / run prediction / detection / whatever:
python eval.py -m /path/to/your/model.pth.tar  -i /path/to/eval/images -o /path/to/output/result


CODE CHANGES / CUSTOMIZATIONS
================================================================================================
1) CONFIGURATIONS
File: config.json
================================================================================================

KEY CHANGEABLE VARIABLES: ALLOWED PARAMETERS
"name"         : #Any string
"cuda"         : true / false
"gpu"          : [0]                                           # A list of gpu index numbers
"dataset"      : "mydataset" / "icdar2015" / "synth800k"       # train.py looks for one of these 3 strings to decide how to load data
"save_freq"    : 1                                             # Any integer, represents how many epochs per model save
"mode"         : "recognition" / "detection" / "united"        # Choose the function of your FOTS code. "United" = "recognition" + "detection"
"monitor"      : "loss"                                        # or some other metric type supported by PyTorch. Represents the metric to monitor how 'good' the model is.
"monitor_mode" : "min" / "max"                                 # Represents the quantity of said "monitor" metric type to save best model. "max" for "accuracy", "min" for "loss"
"keys"         : "alphabet_and_number"                         # Choose a string that contains the characters you want FOTS to learn to recognize from utils/common_str.py / 
                                                               # create your own

#The rest are self-explanatory

================================================================================================
2) GT File-naming code (will affect file retrieval for training)
File: data_loader/datautils.py
================================================================================================

ORIGINAL:
def image_label(txt_root, image_list, img_name, index,
                input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                background_ratio=3. / 8,
                random_rotate_degree=np.arange(-15, 16, 2),
                ):
    """
    get image's corresponding matrix and ground truth
    ??????????,input_size??????,??128?????
    """

    try:
        image_filename = image_list[index]
        cur_img_name = img_name[index]
        cur_img = cv2.imread(image_filename)
        h, w, _ = cur_img.shape

        gt_file_name = 'gt_' + cur_img_name.replace(cur_img_name.split('.')[1], 'txt')
        gt_file_name = os.path.join(txt_root, gt_file_name)
        
        ... ... ...
        
CHANGED:
def image_label(txt_root, image_list, img_name, index,
                input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                background_ratio=3. / 8,
                random_rotate_degree=np.arange(-15, 16, 2),
                ):
    """
    get image's corresponding matrix and ground truth
    ??????????,input_size??????,??128?????
    """

    try:
        image_filename = image_list[index]
        cur_img_name = img_name[index]
        cur_img = cv2.imread(image_filename)
        h, w, _ = cur_img.shape

        gt_file_name = cur_img_name.replace(cur_img_name.split('.')[1], 'txt')
        gt_file_name = os.path.join(txt_root, gt_file_name)
        
        ... ... ...
        
================================================================================================
3) Label-parsing Code (Affects deletion of special characters from labels)
All 4 parts must be changed together, if you wish to change them
File: utils/eval_tools/icdar2015/eval.py
================================================================================================

#i
ORIGINAL:
def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'WORD_SPOTTING': False,
        'MIN_LENGTH_CARE_WORD': 3,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': False,  # Detections must include confidence value. MAP and MAR will be calculated,
        'SPECIAL_CHARACTERS': '!?.:,*"()·[]/\'',
        'ONLY_REMOVE_FIRST_LAST_CHARACTER': True
    }

CHANGED:
def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'WORD_SPOTTING': False,
        'MIN_LENGTH_CARE_WORD': 3,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': False,  # Detections must include confidence value. MAP and MAR will be calculated,
        'SPECIAL_CHARACTERS': '',
        'ONLY_REMOVE_FIRST_LAST_CHARACTER': True
    }
    
#ii
ORIGINAL:
    def transcription_match(transGt, transDet, specialCharacters = '!?.:,*"()·[]/\'',
                            onlyRemoveFirstLastCharacterGT = True):
                            
CHANGED:
    def transcription_match(transGt, transDet, specialCharacters = '',
                            onlyRemoveFirstLastCharacterGT = True):

#iii
ORIGINAL:
    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        """         #SHIFTED COMMENT FROM HERE
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()Â·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')
            
        transcription = transcription.strip()
        
CHANGED:
    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()Â·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')
        """        #TO HERE
        transcription = transcription.strip()
        
#iv
ORIGINAL:
    def include_in_dictionary_transcription(transcription):
        """ 
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        """        #SHIFTED COMMENT FROM HERE
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()Â·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')
            
        transcription = transcription.strip()

        return transcription
        
CHANGED:
    def include_in_dictionary_transcription(transcription):
        """
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = "'!?.:,*\"()Â·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')
        """   #TO HERE
        transcription = transcription.strip()

        return transcription
