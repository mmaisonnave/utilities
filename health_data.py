from dataclasses import dataclass
from enum import Enum
import datetime
from typing import Union
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
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
               self.case_weight is None or \
               self.admit_date is None

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
    def continuos_matrix(admissions: list[Self])->np.ndarray:
        return np.vstack([ admission.continuos_vector for admission in admissions])
    @staticmethod
    def categorical_matrix(admissions: list[Self])->np.ndarray:
        return np.vstack([ admission.categorical_vector for admission in admissions])
    
    @staticmethod
    def create_codes_matrix(admissions: list[Self])->(np.ndarray, sparse._csr.csr_matrix):
        codes = [' '.join(admission.diagnosis.codes) for admission in admissions]
        vectorizer = TfidfVectorizer(use_idf=False).fit(codes)
        print(f'Number of features={len(vectorizer.get_feature_names_out())}')
        return vectorizer.get_feature_names_out(), vectorizer.transform(codes)

        
    @staticmethod
    def continuos_columns() -> list[str]:
        return ['Length of Stay', 'Case Weight', 'CMG', 'Age', 'ALC Days']
    @staticmethod
    def categorical_columns() -> list[str]:
        return ['Male', 
                'Female', 
                'Elective Admission', 
                'New Born Admission', 
                'Urgent Admission', 
                'Transfusion Given', 
                'Level 1 Comorbidity',
                'Level 2 Comorbidity',
                'Level 3 Comorbidity',
                'Level 4 Comorbidity',
                'No Comorbidity',
                'Is ALC',
                'Is central Zone'
                ]
    @property
    def continuos_vector(self: Self,) -> np.ndarray:
        return np.array([self.length_of_stay, 
                        self.case_weight, 
                        self.cmg, 
                        self.age, 
                        self.alc_days,
                        ])

    @property
    def categorical_vector(self: Self,) -> np.ndarray:
        return np.array([1 if self.gender.is_male else 0,
                         1 if self.gender.is_female else 0,
                         1 if self.admit_category==AdmitCategory.ELECTIVE else 0,
                         1 if self.admit_category==AdmitCategory.NEW_BORN else 0,
                         1 if self.admit_category==AdmitCategory.URGENT else 0,
                         1 if self.transfusion_given.received_transfusion else 0,
                         1 if self.comorbidity_level==ComorbidityLevel.LEVEL_1_COMORBIDITY else 0,
                         1 if self.comorbidity_level==ComorbidityLevel.LEVEL_2_COMORBIDITY else 0,
                         1 if self.comorbidity_level==ComorbidityLevel.LEVEL_3_COMORBIDITY else 0,
                         1 if self.comorbidity_level==ComorbidityLevel.LEVEL_4_COMORBIDITY else 0,
                         1 if self.comorbidity_level==ComorbidityLevel.NO_COMORBIDITY else 0,
                         1 if self.alc_days>0 else 0,
                         1 if self.is_central_zone else 0,
                        ])

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


