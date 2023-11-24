from dataclasses import dataclass, fields
from enum import Enum
import datetime
from typing import Union
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from utilities import configuration
import json
from collections import defaultdict

# class syntax
class ReadmissionCode(Enum):
    PLANNED_READMIT = 1
    UNPLANNED_READMIT_0_7 = 2
    UNPLANNED_READMIT_8_28 = 3
    UNPLANNED_FROM_SDS_0_7 = 4
    NEW_ACUTE_PATIENT = 5
    OTHER = 9
    NONE=-1

    # def is_valid_readmit(self, discharge_date: datetime.datetime, readmission_date:datetime.datetime)-> bool:
    #     """        
    #     For readmissions only, gives the lower and upper bound since the last admission (the one that led to the readmission)
    #     Args:
    #         self: 
    #     Returns:
    #         Tuple (int,int) with lower and upper bound.
    #     """
    #     if self==ReadmissionCode.PLANNED_READMIT:
    #        return True
    #     elif self in (ReadmissionCode.UNPLANNED_READMIT_0_7, ReadmissionCode.UNPLANNED_FROM_SDS_0_7):
    #         return 0 <= (readmission_date-discharge_date).days  and (readmission_date-discharge_date).days <= 7
    #     elif self.UNPLANNED_READMIT_8_28:
    #         # This assert doesn't work, some readmissions labeled 8-28 days are less than 8 days
    #         # return 8 <= (readmission_date-discharge_date).days  and (readmission_date-discharge_date).days <= 28
    #         return (readmission_date-discharge_date).days <= 28
        
            
    @staticmethod
    def is_readmit(admission_code:Self):
        """
        Check if the readmission code is one of the one flagging a readmit.

        Args:
            self: 
        Returns:
            True if the readmit code is one flagging a readmit.
        """
        return admission_code in (ReadmissionCode.UNPLANNED_READMIT_8_28, 
                                  ReadmissionCode.UNPLANNED_READMIT_0_7,  
                                  ReadmissionCode.PLANNED_READMIT, 
                                  ReadmissionCode.UNPLANNED_FROM_SDS_0_7)

# class syntax
class ComorbidityLevel(Enum):
    NO_COMORBIDITY = 0
    LEVEL_1_COMORBIDITY = 1
    LEVEL_2_COMORBIDITY = 2
    LEVEL_3_COMORBIDITY = 3
    LEVEL_4_COMORBIDITY = 4
    NOT_APPLICABLE = 8
    NONE=-1


# # class syntax
# class CentralZoneStatus(Enum):
#     CENTRAL_ZONE = 1
#     NON_CENTRAL_ZONE = 2

# class syntax
class TransfusionGiven(Enum):
    NO = 0
    YES = 1
    NONE=-1
    @property
    def received_transfusion(self: Self,)->bool:
        return self == TransfusionGiven.YES


class AdmitCategory(Enum):
    ELECTIVE = 1
    NEW_BORN = 2
    CADAVER = 3
    STILLBORN = 5
    URGENT = 6 
    NONE = -1


class Gender(Enum):
    FEMALE = 1
    MALE = 2
    UNDIFFERENTIATED = 3
    OTHER = 4
    NONE = -1

    @property
    def is_male(self:Self, )->bool:
        return self == Gender.MALE
    
    @property
    def is_female(self:Self, )->bool:
        return self == Gender.FEMALE


@dataclass
class Diagnosis:
    codes: str
    texts: str
    types: str    
    
    # def __post_init__(self):
    #     if not (len(self.codes)==len(self.texts) and len(self.texts)==len(self.types)):
    #         print(f'len codes: {self.codes}')
    #         print(f'len texts: {self.texts}')
    #         print(f'len types: {self.types}')
    #     assert len(self.codes)==len(self.texts) and len(self.texts)==len(self.types)

@dataclass
class Admission:
    admit_id: int
    code: Union[int,None]
    institution_number: int
    admit_date: Union[datetime.datetime,None]
    discharge_date: datetime.datetime
    readmission_code: ReadmissionCode
    age: int
    gender: Gender
    mrdx: str
    postal_code: str
    diagnosis: Diagnosis
    intervention_code:list
    px_long_text:list
    admit_category: AdmitCategory
    transfusion_given: TransfusionGiven
    main_pt_service:str
    cmg: Union[float,None]
    comorbidity_level:ComorbidityLevel
    case_weight: Union[float,None]
    alc_days: int
    acute_days: int
    institution_to: str
    institution_from: str
    institution_type: str
    discharge_unit:str
    is_central_zone: bool
    readmission: Self

    @property
    def has_missing(self:Self,)->bool:
        """
        Check if some of the attributes are None (Except the enums).
        This methods checks:
            - HCN code
            - CMG 
            - Case Weight and
            - Admit date
        Returns:
            True if any of those four attributes are None
        """
        return self.code is None or \
               self.cmg is None or \
               np.isnan(self.cmg) or \
               self.case_weight is None or \
               np.isnan(self.case_weight) or \
               self.admit_date is None or \
               self.readmission_code == ReadmissionCode.NONE or \
               self.gender == Gender.NONE or \
               self.admit_category == AdmitCategory.NONE or \
               self.main_pt_service is None or \
               self.mrdx is None or \
               self.transfusion_given == TransfusionGiven.NONE 
               

    def __iter__(self: Self):
        return ((field.name, getattr(self, field.name)) for field in fields(self))
    
    def __post_init__(self):
        if not self.admit_date is None:
            assert self.admit_date <= self.discharge_date

        if self.admit_category != AdmitCategory.NEW_BORN:
            assert 0<=self.age
        else: # NEW BORN
            assert -1<=self.age

    @staticmethod
    def from_dict_data(admit_id:int, admission:dict) -> Self:
        # Readmission code
        readmission_code = ReadmissionCode(int(admission['Readmission Code'][0])) if not admission['Readmission Code'] is None else ReadmissionCode.NONE

        #Diagnosis 
        diagnosis = Diagnosis(codes=admission['Diagnosis Code'], texts=admission['Diagnosis Long Text'] , types=admission['Diagnosis Type'])

        # Admit Category
        if admission['Admit Category'] is None:
            admit_category = AdmitCategory.NONE
        elif 'Elective' in admission['Admit Category']:
            admit_category = AdmitCategory.ELECTIVE
        elif 'Newborn' in admission['Admit Category']:
            admit_category = AdmitCategory.NEW_BORN
        elif 'Cadaver' in admission['Admit Category']:
            admit_category = AdmitCategory.CADAVER
        elif 'Stillborn' in admission['Admit Category']:
            admit_category = AdmitCategory.STILLBORN
        elif 'urgent' in admission['Admit Category']:
            admit_category = AdmitCategory.URGENT

        if admission['Transfusion Given'] is None:
            transfusion = TransfusionGiven.NONE
        elif admission['Transfusion Given']=='Yes':
            transfusion = TransfusionGiven.YES
        elif admission['Transfusion Given']=='No':
            transfusion = TransfusionGiven.NO

        
        if admission['Gender']=='Male':
            gender=Gender.MALE
        elif admission['Gender']=='Female':
            gender=Gender.FEMALE
        elif admission['Gender']=='Other (transsexu':
            gender=Gender.OTHER
        elif admission['Gender']=='Undifferentiated':
            gender=Gender.UNDIFFERENTIATED
        elif admission['Gender'] is None:
            gender=Gender.NONE

        # Readmission code
        comorbidity_level = ComorbidityLevel(int(admission['Comorbidity Level'][0])) if not admission['Comorbidity Level'] is None else ComorbidityLevel.NONE
        admission = Admission(admit_id=int(admit_id),
                        code=int(admission['HCN code']) if not admission['HCN code']is None else None,
                        institution_number = int(admission['Institution Number']),
                        admit_date = datetime.datetime.fromisoformat(admission['Admit Date']) if not admission['Admit Date'] is None else None,
                        discharge_date = datetime.datetime.fromisoformat(admission['Discharge Date']),
                        readmission_code = readmission_code,
                        age = int(admission['Patient Age']),
                        gender = gender,
                        mrdx = str(admission['MRDx']),
                        postal_code = str(admission['Postal Code']),
                        diagnosis = diagnosis,
                        intervention_code = admission['Intervention Code'],
                        px_long_text = admission['Px Long Text'],
                        admit_category = admit_category,
                        transfusion_given = transfusion,
                        main_pt_service = admission['Main Pt Service'],
                        cmg = float(admission['CMG']) if not admission['CMG'] is None else None,
                        comorbidity_level = comorbidity_level,
                        case_weight = float(admission['Case Weight']) if not admission['Case Weight'] is None else None,
                        alc_days = int(admission['ALC Days']),
                        acute_days = int(admission['Acute Days']),
                        institution_to = admission['Institution To'],
                        institution_from = admission['Institution From'],
                        institution_type = admission['Institution Type'],
                        discharge_unit = admission['Discharge Nurse Unit'],
                        is_central_zone = admission['CZ Status']=='cz',
                        readmission=None
                        )
        return admission
    

    def __repr__(self: Self,)->str:
        repr_ = f"<Admission Patient_code='{self.code}' "\
            f"admit='{self.admit_date.date()}' "\
                f"discharged='{self.discharge_date.date()}' "\
                    f"Age='{self.age}' gender='{self.gender}' ALC_days='{self.alc_days}' acute_days='{self.acute_days}' readmited=No>"
        if not self.readmission is None:
            repr_ = repr_[:-13] + f'readmited({self.readmission.admit_date.date()},{self.readmission.discharge_date.date()},{self.readmission.readmission_code})>'
        return repr_

    
    @staticmethod
    def diagnosis_codes_features(admissions: list[Self], vocabulary=None, use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix):
        codes = [' '.join(admission.diagnosis.codes) for admission in admissions]
        if vocabulary is None:
            vectorizer = TfidfVectorizer(use_idf=use_idf).fit(codes)
        else:
            vectorizer = TfidfVectorizer(use_idf=use_idf, vocabulary=vocabulary).fit(codes)

        return vectorizer.get_feature_names_out(), vectorizer.transform(codes)
    
    @staticmethod
    def intervention_codes_features(admissions: list[Self], vocabulary=None, use_idf:bool = False)->(np.ndarray, sparse._csr.csr_matrix):
        codes = [' '.join(admission.intervention_code) for admission in admissions]
        if vocabulary is None:
            vectorizer = TfidfVectorizer(use_idf=use_idf).fit(codes)
        else:
            vectorizer = TfidfVectorizer(use_idf=use_idf, vocabulary=vocabulary).fit(codes)
        return vectorizer.get_feature_names_out(), vectorizer.transform(codes)

    @staticmethod
    def categorical_features(admissions: list[Self],main_pt_services_list=None) -> pd.DataFrame:
        columns = ['male', 
                   'female', 
                   'transfusion given', 
                   'is alc',
                   'is central zone',
                   'elective admission',
                   'new born admission',
                   'urgent admission',
                   'level 1 comorbidity',
                   'level 2 comorbidity',
                   'level 3 comorbidity',
                   'level 4 comorbidity',
                   ]
        if main_pt_services_list is None:
            main_pt_services_list = list(set([admission.main_pt_service for admission in admissions]))[:-1]
        service2idx = dict([(service,ix+len(columns)) for ix,service in enumerate(main_pt_services_list)])
        columns = columns + main_pt_services_list

        vectors = []
        for admission in admissions:
            vector = [1 if admission.gender.is_male else 0,
                     1 if admission.gender.is_female else 0,
                     1 if admission.transfusion_given.received_transfusion else 0,
                     1 if admission.alc_days > 0 else 0,
                     1 if admission.is_central_zone else 0,
                     1 if admission.admit_category==AdmitCategory.ELECTIVE else 0,
                     1 if admission.admit_category==AdmitCategory.NEW_BORN else 0,
                     1 if admission.admit_category==AdmitCategory.URGENT else 0,
                    #  1 if admission.comorbidity_level==ComorbidityLevel.NO_COMORBIDITY else 0,      # 8
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_1_COMORBIDITY else 0, # 8
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_2_COMORBIDITY else 0, # 9
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_3_COMORBIDITY else 0, # 10
                     1 if admission.comorbidity_level==ComorbidityLevel.LEVEL_4_COMORBIDITY else 0, # 11
                    ]
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_2_COMORBIDITY:
                assert vector[8]==0 and vector[9] == 1 and vector[10] == 0 and vector[11] == 0
                vector[8]=1
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_3_COMORBIDITY:
                assert vector[8]==0 and vector[9] == 0 and vector[10] == 1 and vector[11] == 0
                vector[8]=1
                vector[9]=1
            if admission.comorbidity_level==ComorbidityLevel.LEVEL_4_COMORBIDITY:
                assert vector[8]==0 and vector[9] == 0 and vector[10] == 0 and vector[11] == 1
                vector[8]=1
                vector[9]=1
                vector[10]=1
            vector = vector + [0]*len(main_pt_services_list)
            if admission.main_pt_service in service2idx:
                vector[service2idx[admission.main_pt_service]]=1
            vectors.append(vector)

        return pd.DataFrame(vectors, columns=columns),main_pt_services_list

    @property
    def is_valid_training_instance(self:Self)->bool:
        return self.is_valid_testing_instance and not self.has_missing
    
    @property
    def is_valid_testing_instance(self:Self)->bool:
        return self.admit_category != AdmitCategory.CADAVER and \
                self.admit_category != AdmitCategory.STILLBORN and \
                not self.code is None

    @staticmethod
    def numerical_features(admissions: list[Self],) -> pd.DataFrame:
        fields = ['age', 'cmg', 'case_weight', 'acute_days', 'alc_days']
        # assert all([admission.is for admission in admissions])
        vectors = []
        for admission in admissions:
            vectors.append([getattr(admission, field) for field in fields])
        matrix = np.vstack(vectors)

        df =  pd.DataFrame(matrix, columns=fields)

        # Missing from training are removed, missing from testing are fixed. 
        # Should not be na values.
        assert df.dropna().shape[0]==df.shape[0]

        return df

      
    @staticmethod 
    def get_y(admissions: list[Self])->np.ndarray:
        return np.array([1 if admission.has_readmission and \
                     admission.readmission.readmission_code!=ReadmissionCode.PLANNED_READMIT else 0 \
                     for admission in admissions])

    @property
    def has_readmission(self: Self,)->bool:
        return not self.readmission is None

    @property
    def length_of_stay(self: Self)->int:
        los = None
        if not self.admit_date is None:
            los = (self.discharge_date - self.admit_date).days
        return los
    
    def is_valid_readmission(self, readmission: Self)->bool:
        """
        Check if the readmission is valid. The readmission is valid if 
            - the readmission is at most 30 days later than the original admission (self) and
            - the readmission is as a readmission_core that indicates is a readmission and not a 
              first admission (or others).

        Args:
            readmission:The readmission to check if it is a valid readmission to the admission that receives the msg
        Returns:
            True if the `readmission` is a valid readmission for the admission that received the msg.
        """
        return (readmission.admit_date - self.discharge_date).days<=30 and \
            ReadmissionCode.is_readmit(readmission.readmission_code)
    
    def add_readmission(self, readmission: Self):
        """
        Adds the (re)admission sent as parameter as the readmission to the admission that receives the msg.
        Requires the readmission is valid. 

        Args:
            readmission: Admission to add as readmission.
        """
        assert self.is_valid_readmission(readmission)
        self.readmission = readmission

    # def fix_missing(self: Self, )-> Self:
    #     rng = np.random.default_rng(seed=5348363479653547918)

    @staticmethod
    def get_training_testing_data(filtering=True)->list[Self]:
        rng = np.random.default_rng(seed=5348363479653547918)
        config = configuration.get_config()

        # ---------- ---------- ---------- ---------- 
        # Retriving train testing data from JSON file
        # ---------- ---------- ---------- ---------- 
        f = open(config['train_val_json'])
        train_val_data = json.load(f)

        # ---------- ---------- ---------- ---------- 
        # Converting JSON to DataClasses
        # ---------- ---------- ---------- ---------- 
        all_admissions = []
        for ix in train_val_data:
            all_admissions.append(
                Admission.from_dict_data(admit_id=int(ix), admission=train_val_data[ix])
                )
            


        # ---------- ---------- ---------- ---------- 
        # Dictionary organizing data by patient
        # ---------- ---------- ---------- ---------- 
        patient2admissions = defaultdict(list)
        for admission in all_admissions:
            code = admission.code
            patient2admissions[code].append(admission)

        # ---------- ---------- ---------- ---------- 
        # Ordering patient list by discharge date (from back )
        # ---------- ---------- ---------- ---------- 
        for patient_code in patient2admissions:
            admissions_list = patient2admissions[patient_code]
            admissions_list = sorted(admissions_list, key=lambda admission: admission.discharge_date, reverse=False)
            assert all([admissions_list[i].discharge_date <= admissions_list[i+1].discharge_date for i in range(len(admissions_list)-1)])
            patient2admissions[patient_code] = admissions_list

        patient_count=0
        valid_readmission_count=0
        for patient_code in patient2admissions:
            patient_admissions = patient2admissions[patient_code]
            ix = 0 
            while ix < len(patient_admissions):
                readmission_code = patient_admissions[ix].readmission_code
                if ReadmissionCode.is_readmit(readmission_code):
                    # Either is not the first admission (ix>0) or 
                    # we don't have the patient previous admition (readmission close to begining of dataset) (admit-(2015-01-01))<28 days
                    # assert ix>0 or (patient_admissions[ix].admit_date - datetime.datetime.fromisoformat('2015-01-01')).days<365
                    if ix>0 and  patient_admissions[ix-1].is_valid_readmission(patient_admissions[ix]):
                        patient_admissions[ix-1].add_readmission(patient_admissions[ix])
                        valid_readmission_count+=1
                ix+=1
            patient_count+=1

        train_indexes = rng.choice(range(len(all_admissions)),size=int(0.8*len(all_admissions)), replace=False)

        # Checking that every time I am getting the same training instances ( and validation instances)
        assert all(train_indexes[:3] ==np.array([478898, 46409, 322969]))
        assert all(train_indexes[-3:] ==np.array([415014, 330673, 338415]))
        assert hash(tuple(train_indexes))==2028319680436964623

        train_indexes = set(train_indexes)

        train = [admission for ix, admission in enumerate(all_admissions) if ix in train_indexes ]
        testing = [admission for ix, admission in enumerate(all_admissions) if not ix in train_indexes ]


        # ---------- ---------- ---------- ----------
        # Filtering instances with missing values
        # ---------- ---------- ---------- ----------
        # Remove from training instances with missing values or with admit category in {CADAVER, STILLBORN}
        if filtering:
            print(f'Training instances before filtering: {len(train)}')
            train = list(filter(lambda admission: admission.is_valid_training_instance, train))
            print(f'Training instances after filtering:  {len(train)}')

            # Remove from testing instances without patient code and admit category in {CADAVER, STILLBORN}
            print(f'Testomg instances before filtering:  {len(testing)}')
            testing = list(filter(lambda admission: admission.is_valid_testing_instance , testing))
            print(f'Testomg instances after filtering:   {len(testing)}')


        return train, testing

    def fix_missings(self: Self, training: list[Self]):
        rng = np.random.default_rng(seed=5348363479653547918)
        assert not self.code is None, 'Cannot fix an entry without code (cannot recover target variable without it).'

        if self.admit_date is None:
            avg_los = np.average([admission.length_of_stay for admission in training])
            std_los = np.std([admission.length_of_stay for admission in training])
            los = int(rng.normal(loc=avg_los, scale=std_los, size=1)[0])

            self.admit_date = self.discharge_date - datetime.timedelta(days=los)

        if self.case_weight is None or np.isnan(self.case_weight):
            avg_case_weight =  np.average([admission.case_weight for admission in training])
            std_case_weight =  np.std([admission.case_weight for admission in training])

            self.case_weight = rng.normal(loc=avg_case_weight, scale=std_case_weight, size=1)[0]

        if self.gender  == Gender.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.gender = training[ix].gender

        if self.admit_category == AdmitCategory.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.admit_category = training[ix].admit_category
            
        if self.readmission_code == ReadmissionCode.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.readmission_code = training[ix].readmission_code

        if self.transfusion_given == TransfusionGiven.NONE:
            ix = rng.choice(a=range(len(training)), size=1)[0]
            self.transfusion_given = training[ix].transfusion_given

        if self.cmg is None or np.isnan(self.cmg):
            new_cmb = rng.uniform(low=min([admission.cmg for admission in training]), 
                                high=max([admission.cmg for admission in training]), 
                                size=1)[0]
            self.cmg = new_cmb

        if self.main_pt_service is None:
            self.main_pt_service = '<NONE>'

        if self.mrdx is None:
            self.mrdx = '<NONE>'