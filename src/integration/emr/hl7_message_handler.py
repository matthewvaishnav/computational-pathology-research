"""
HL7 Message Handler for EMR Integration

Comprehensive HL7 v2.x message processing for EMR integrations including
parsing, validation, transformation, and routing of clinical messages.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from hl7apy import core, parse_message
from hl7apy.core import Component, Field, Message, Segment
from hl7apy.exceptions import ChildNotFound, InvalidName


class HL7MessageType(Enum):
    """HL7 message types"""

    ADT_A01 = "ADT^A01"  # Admit/Visit Notification
    ADT_A02 = "ADT^A02"  # Transfer Patient
    ADT_A03 = "ADT^A03"  # Discharge/End Visit
    ADT_A04 = "ADT^A04"  # Register Patient
    ADT_A08 = "ADT^A08"  # Update Patient Information
    ORM_O01 = "ORM^O01"  # Order Message
    ORU_R01 = "ORU^R01"  # Observation Result
    SIU_S12 = "SIU^S12"  # Notification of New Appointment Booking
    MDM_T02 = "MDM_T02"  # Original Document Notification
    ACK = "ACK"  # General Acknowledgment


class HL7ProcessingStatus(Enum):
    """HL7 message processing status"""

    RECEIVED = "received"
    PARSING = "parsing"
    VALIDATED = "validated"
    PROCESSED = "processed"
    ROUTED = "routed"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    REJECTED = "rejected"


class HL7AckCode(Enum):
    """HL7 acknowledgment codes"""

    AA = "AA"  # Application Accept
    AE = "AE"  # Application Error
    AR = "AR"  # Application Reject
    CA = "CA"  # Conditional Accept
    CE = "CE"  # Conditional Error
    CR = "CR"  # Conditional Reject


@dataclass
class HL7MessageInfo:
    """HL7 message information"""

    message_id: str
    message_type: HL7MessageType
    sending_application: str
    sending_facility: str
    receiving_application: str
    receiving_facility: str
    timestamp: datetime
    control_id: str
    processing_id: str
    version_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["message_type"] = self.message_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class HL7Patient:
    """HL7 patient information"""

    patient_id: str
    patient_id_list: List[str]
    patient_name: str
    date_of_birth: Optional[datetime]
    gender: Optional[str]
    race: Optional[str]
    address: Optional[str]
    phone: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.date_of_birth:
            data["date_of_birth"] = self.date_of_birth.isoformat()
        return data


@dataclass
class HL7Order:
    """HL7 order information"""

    order_control: str
    placer_order_number: str
    filler_order_number: str
    universal_service_id: str
    priority: Optional[str]
    requested_datetime: Optional[datetime]
    ordering_provider: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.requested_datetime:
            data["requested_datetime"] = self.requested_datetime.isoformat()
        return data


@dataclass
class HL7Observation:
    """HL7 observation/result information"""

    set_id: str
    value_type: str
    observation_id: str
    observation_value: str
    units: Optional[str]
    reference_range: Optional[str]
    abnormal_flags: Optional[str]
    observation_result_status: str
    observation_datetime: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.observation_datetime:
            data["observation_datetime"] = self.observation_datetime.isoformat()
        return data


class HL7MessageHandler:
    """
    Comprehensive HL7 message handler

    Features:
    - HL7 v2.x message parsing and validation
    - Message type routing and processing
    - Acknowledgment generation
    - Error handling and logging
    - Message transformation and mapping
    - Integration with EMR plugins
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize HL7 message handler"""
        self.config = config

        # Handler settings
        self.receiving_application = config.get("receiving_application", "AI_PATHOLOGY")
        self.receiving_facility = config.get("receiving_facility", "AI_LAB")

        # Message processing
        self.message_handlers = {}
        self.transformation_rules = {}
        self.validation_rules = {}

        # Error handling
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 60)

        # Message storage
        self.message_store = config.get("message_store", "memory")  # memory, file, database
        self.stored_messages = {}

        # Acknowledgment settings
        self.auto_ack = config.get("auto_ack", True)
        self.ack_timeout = config.get("ack_timeout", 30)

        self.logger = logging.getLogger(__name__)

    def register_message_handler(self, message_type: HL7MessageType, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler

    def register_transformation_rule(self, message_type: HL7MessageType, rule: Callable):
        """Register transformation rule for message type"""
        self.transformation_rules[message_type] = rule

    def register_validation_rule(self, message_type: HL7MessageType, rule: Callable):
        """Register validation rule for message type"""
        self.validation_rules[message_type] = rule

    async def process_message(self, hl7_message: str, source: str = "unknown") -> Dict[str, Any]:
        """Process incoming HL7 message"""
        processing_result = {
            "status": HL7ProcessingStatus.RECEIVED.value,
            "message_id": None,
            "ack_message": None,
            "errors": [],
            "warnings": [],
            "processed_data": None,
        }

        try:
            # Parse message
            self.logger.info(f"Processing HL7 message from {source}")
            processing_result["status"] = HL7ProcessingStatus.PARSING.value

            parsed_message = self._parse_hl7_message(hl7_message)
            if not parsed_message:
                raise ValueError("Failed to parse HL7 message")

            message_info = self._extract_message_info(parsed_message)
            processing_result["message_id"] = message_info.control_id

            # Store message
            await self._store_message(message_info.control_id, hl7_message, message_info)

            # Validate message
            processing_result["status"] = HL7ProcessingStatus.VALIDATED.value
            validation_result = await self._validate_message(parsed_message, message_info)

            if not validation_result["valid"]:
                processing_result["errors"].extend(validation_result["errors"])
                processing_result["status"] = HL7ProcessingStatus.REJECTED.value

                # Generate error ACK
                if self.auto_ack:
                    ack_message = self._generate_ack_message(
                        message_info, HL7AckCode.AE, validation_result["errors"][0]
                    )
                    processing_result["ack_message"] = ack_message

                return processing_result

            # Transform message if needed
            transformed_data = await self._transform_message(parsed_message, message_info)

            # Process message
            processing_result["status"] = HL7ProcessingStatus.PROCESSED.value
            processed_data = await self._route_and_process_message(
                parsed_message, message_info, transformed_data
            )
            processing_result["processed_data"] = processed_data

            # Generate success ACK
            processing_result["status"] = HL7ProcessingStatus.ACKNOWLEDGED.value
            if self.auto_ack:
                ack_message = self._generate_ack_message(message_info, HL7AckCode.AA)
                processing_result["ack_message"] = ack_message

            self.logger.info(f"Successfully processed HL7 message {message_info.control_id}")

        except Exception as e:
            self.logger.error(f"Error processing HL7 message: {e}")
            processing_result["status"] = HL7ProcessingStatus.FAILED.value
            processing_result["errors"].append(str(e))

            # Generate error ACK if we have message info
            if self.auto_ack and "message_id" in locals():
                try:
                    ack_message = self._generate_ack_message(message_info, HL7AckCode.AE, str(e))
                    processing_result["ack_message"] = ack_message
                except Exception as ack_error:
                    # Log ACK generation failure
                    logger.error(
                        f"Failed to generate ACK message: error_code=ACK_GEN_FAILED, "
                        f"original_error={type(e).__name__}"
                    )

        return processing_result

    def _parse_hl7_message(self, hl7_message: str) -> Optional[Message]:
        """Parse HL7 message string"""
        try:
            # Clean message
            cleaned_message = hl7_message.strip()

            # Replace line endings
            cleaned_message = cleaned_message.replace("\n", "\r")

            # Parse with hl7apy
            parsed_message = parse_message(cleaned_message)
            return parsed_message

        except Exception as e:
            self.logger.error(f"HL7 parsing error: {e}")
            return None

    def _extract_message_info(self, parsed_message: Message) -> HL7MessageInfo:
        """Extract message header information"""
        try:
            msh = parsed_message.msh

            # Extract message type
            message_type_field = msh.msh_9.value
            message_type = HL7MessageType(message_type_field)

            # Extract other header fields
            return HL7MessageInfo(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                sending_application=msh.msh_3.value,
                sending_facility=msh.msh_4.value,
                receiving_application=msh.msh_5.value,
                receiving_facility=msh.msh_6.value,
                timestamp=self._parse_hl7_datetime(msh.msh_7.value),
                control_id=msh.msh_10.value,
                processing_id=msh.msh_11.value,
                version_id=msh.msh_12.value,
            )

        except Exception as e:
            self.logger.error(f"Error extracting message info: {e}")
            raise

    def _parse_hl7_datetime(self, hl7_datetime: str) -> datetime:
        """Parse HL7 datetime format"""
        try:
            # HL7 datetime format: YYYYMMDDHHMMSS
            if len(hl7_datetime) >= 8:
                if len(hl7_datetime) == 8:
                    # Date only
                    return datetime.strptime(hl7_datetime, "%Y%m%d")
                elif len(hl7_datetime) >= 14:
                    # Full datetime
                    return datetime.strptime(hl7_datetime[:14], "%Y%m%d%H%M%S")
                else:
                    # Partial datetime
                    padded = hl7_datetime.ljust(14, "0")
                    return datetime.strptime(padded[:14], "%Y%m%d%H%M%S")

            return datetime.now()

        except Exception as e:
            self.logger.error(f"Error parsing HL7 datetime {hl7_datetime}: {e}")
            return datetime.now()

    async def _validate_message(
        self, parsed_message: Message, message_info: HL7MessageInfo
    ) -> Dict[str, Any]:
        """Validate HL7 message"""
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Basic structure validation
            if not hasattr(parsed_message, "msh"):
                validation_result["valid"] = False
                validation_result["errors"].append("Missing MSH segment")
                return validation_result

            # Message type specific validation
            if message_info.message_type in self.validation_rules:
                rule_result = await self.validation_rules[message_info.message_type](parsed_message)
                validation_result["valid"] = validation_result["valid"] and rule_result.get(
                    "valid", True
                )
                validation_result["errors"].extend(rule_result.get("errors", []))
                validation_result["warnings"].extend(rule_result.get("warnings", []))

            # Required fields validation
            required_fields = self._get_required_fields(message_info.message_type)
            for field_path in required_fields:
                if not self._check_field_exists(parsed_message, field_path):
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required field: {field_path}")

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")

        return validation_result

    def _get_required_fields(self, message_type: HL7MessageType) -> List[str]:
        """Get required fields for message type"""
        required_fields = {
            HL7MessageType.ADT_A01: ["PID.3", "PID.5"],
            HL7MessageType.ADT_A04: ["PID.3", "PID.5"],
            HL7MessageType.ORM_O01: ["PID.3", "ORC.1", "OBR.1"],
            HL7MessageType.ORU_R01: ["PID.3", "OBR.1", "OBX.1"],
        }

        return required_fields.get(message_type, [])

    def _check_field_exists(self, parsed_message: Message, field_path: str) -> bool:
        """Check if field exists in message"""
        try:
            parts = field_path.split(".")
            segment_name = parts[0]
            field_num = int(parts[1]) if len(parts) > 1 else None

            # Get segment
            if hasattr(parsed_message, segment_name.lower()):
                segment = getattr(parsed_message, segment_name.lower())

                if field_num:
                    # Check specific field
                    field_name = f"{segment_name.lower()}_{field_num}"
                    if hasattr(segment, field_name):
                        field = getattr(segment, field_name)
                        return field.value is not None and field.value != ""

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking field {field_path}: {e}")
            return False

    async def _transform_message(
        self, parsed_message: Message, message_info: HL7MessageInfo
    ) -> Dict[str, Any]:
        """Transform HL7 message to internal format"""
        try:
            # Apply transformation rules if available
            if message_info.message_type in self.transformation_rules:
                return await self.transformation_rules[message_info.message_type](parsed_message)

            # Default transformation based on message type
            if message_info.message_type in [
                HL7MessageType.ADT_A01,
                HL7MessageType.ADT_A04,
                HL7MessageType.ADT_A08,
            ]:
                return self._transform_adt_message(parsed_message)
            elif message_info.message_type == HL7MessageType.ORM_O01:
                return self._transform_orm_message(parsed_message)
            elif message_info.message_type == HL7MessageType.ORU_R01:
                return self._transform_oru_message(parsed_message)

            return {}

        except Exception as e:
            self.logger.error(f"Error transforming message: {e}")
            return {}

    def _transform_adt_message(self, parsed_message: Message) -> Dict[str, Any]:
        """Transform ADT message"""
        try:
            # Extract patient information
            pid = parsed_message.pid

            patient_data = HL7Patient(
                patient_id=pid.pid_3.value if hasattr(pid, "pid_3") else "",
                patient_id_list=[pid.pid_3.value] if hasattr(pid, "pid_3") else [],
                patient_name=pid.pid_5.value if hasattr(pid, "pid_5") else "",
                date_of_birth=(
                    self._parse_hl7_datetime(pid.pid_7.value)
                    if hasattr(pid, "pid_7") and pid.pid_7.value
                    else None
                ),
                gender=pid.pid_8.value if hasattr(pid, "pid_8") else None,
                race=pid.pid_10.value if hasattr(pid, "pid_10") else None,
                address=pid.pid_11.value if hasattr(pid, "pid_11") else None,
                phone=pid.pid_13.value if hasattr(pid, "pid_13") else None,
            )

            return {"message_type": "patient_update", "patient": patient_data.to_dict()}

        except Exception as e:
            self.logger.error(f"Error transforming ADT message: {e}")
            return {}

    def _transform_orm_message(self, parsed_message: Message) -> Dict[str, Any]:
        """Transform ORM (Order) message"""
        try:
            # Extract patient information
            pid = parsed_message.pid
            patient_data = self._extract_patient_from_pid(pid)

            # Extract order information
            orc = parsed_message.orc
            obr = parsed_message.obr

            order_data = HL7Order(
                order_control=orc.orc_1.value if hasattr(orc, "orc_1") else "",
                placer_order_number=orc.orc_2.value if hasattr(orc, "orc_2") else "",
                filler_order_number=orc.orc_3.value if hasattr(orc, "orc_3") else "",
                universal_service_id=obr.obr_4.value if hasattr(obr, "obr_4") else "",
                priority=obr.obr_5.value if hasattr(obr, "obr_5") else None,
                requested_datetime=(
                    self._parse_hl7_datetime(obr.obr_6.value)
                    if hasattr(obr, "obr_6") and obr.obr_6.value
                    else None
                ),
                ordering_provider=obr.obr_16.value if hasattr(obr, "obr_16") else None,
            )

            return {
                "message_type": "new_order",
                "patient": patient_data.to_dict(),
                "order": order_data.to_dict(),
            }

        except Exception as e:
            self.logger.error(f"Error transforming ORM message: {e}")
            return {}

    def _transform_oru_message(self, parsed_message: Message) -> Dict[str, Any]:
        """Transform ORU (Result) message"""
        try:
            # Extract patient information
            pid = parsed_message.pid
            patient_data = self._extract_patient_from_pid(pid)

            # Extract observation results
            observations = []

            # Handle multiple OBX segments
            if hasattr(parsed_message, "obx"):
                obx_segments = (
                    parsed_message.obx
                    if isinstance(parsed_message.obx, list)
                    else [parsed_message.obx]
                )

                for obx in obx_segments:
                    observation = HL7Observation(
                        set_id=obx.obx_1.value if hasattr(obx, "obx_1") else "",
                        value_type=obx.obx_2.value if hasattr(obx, "obx_2") else "",
                        observation_id=obx.obx_3.value if hasattr(obx, "obx_3") else "",
                        observation_value=obx.obx_5.value if hasattr(obx, "obx_5") else "",
                        units=obx.obx_6.value if hasattr(obx, "obx_6") else None,
                        reference_range=obx.obx_7.value if hasattr(obx, "obx_7") else None,
                        abnormal_flags=obx.obx_8.value if hasattr(obx, "obx_8") else None,
                        observation_result_status=(
                            obx.obx_11.value if hasattr(obx, "obx_11") else ""
                        ),
                        observation_datetime=(
                            self._parse_hl7_datetime(obx.obx_14.value)
                            if hasattr(obx, "obx_14") and obx.obx_14.value
                            else None
                        ),
                    )
                    observations.append(observation.to_dict())

            return {
                "message_type": "result_report",
                "patient": patient_data.to_dict(),
                "observations": observations,
            }

        except Exception as e:
            self.logger.error(f"Error transforming ORU message: {e}")
            return {}

    def _extract_patient_from_pid(self, pid: Segment) -> HL7Patient:
        """Extract patient data from PID segment"""
        return HL7Patient(
            patient_id=pid.pid_3.value if hasattr(pid, "pid_3") else "",
            patient_id_list=[pid.pid_3.value] if hasattr(pid, "pid_3") else [],
            patient_name=pid.pid_5.value if hasattr(pid, "pid_5") else "",
            date_of_birth=(
                self._parse_hl7_datetime(pid.pid_7.value)
                if hasattr(pid, "pid_7") and pid.pid_7.value
                else None
            ),
            gender=pid.pid_8.value if hasattr(pid, "pid_8") else None,
            race=pid.pid_10.value if hasattr(pid, "pid_10") else None,
            address=pid.pid_11.value if hasattr(pid, "pid_11") else None,
            phone=pid.pid_13.value if hasattr(pid, "pid_13") else None,
        )

    async def _route_and_process_message(
        self,
        parsed_message: Message,
        message_info: HL7MessageInfo,
        transformed_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route and process message based on type"""
        try:
            # Route to specific handler if available
            if message_info.message_type in self.message_handlers:
                return await self.message_handlers[message_info.message_type](
                    parsed_message, transformed_data
                )

            # Default processing
            return {
                "processed": True,
                "message_type": message_info.message_type.value,
                "data": transformed_data,
            }

        except Exception as e:
            self.logger.error(f"Error routing/processing message: {e}")
            raise

    def _generate_ack_message(
        self,
        message_info: HL7MessageInfo,
        ack_code: HL7AckCode,
        error_message: Optional[str] = None,
    ) -> str:
        """Generate HL7 ACK message"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ack_control_id = str(uuid.uuid4())[:10]

            # MSH segment
            msh = (
                f"MSH|^~\\&|{self.receiving_application}|{self.receiving_facility}|"
                f"{message_info.sending_application}|{message_info.sending_facility}|"
                f"{timestamp}||ACK|{ack_control_id}|P|2.5"
            )

            # MSA segment
            msa = f"MSA|{ack_code.value}|{message_info.control_id}"
            if error_message:
                msa += f"|{error_message}"

            return f"{msh}\r{msa}"

        except Exception as e:
            self.logger.error(f"Error generating ACK message: {e}")
            return ""

    async def _store_message(self, message_id: str, raw_message: str, message_info: HL7MessageInfo):
        """Store message for audit/replay"""
        try:
            if self.message_store == "memory":
                self.stored_messages[message_id] = {
                    "raw_message": raw_message,
                    "message_info": message_info.to_dict(),
                    "stored_at": datetime.now().isoformat(),
                }
            elif self.message_store == "file":
                # Store to file system
                storage_dir = Path("data/hl7_messages")
                storage_dir.mkdir(parents=True, exist_ok=True)

                message_file = storage_dir / f"{message_id}.json"
                with open(message_file, "w") as f:
                    json.dump(
                        {
                            "raw_message": raw_message,
                            "message_info": message_info.to_dict(),
                            "stored_at": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                    )

        except Exception as e:
            self.logger.error(f"Error storing message: {e}")

    def validate_hl7_structure(self, hl7_message: str) -> Dict[str, Any]:
        """Validate HL7 message structure"""
        try:
            # Basic format validation
            if not hl7_message.startswith("MSH"):
                return {"valid": False, "error": "Message must start with MSH segment"}

            # Parse message
            parsed_message = self._parse_hl7_message(hl7_message)
            if not parsed_message:
                return {"valid": False, "error": "Failed to parse HL7 message"}

            # Extract message info
            message_info = self._extract_message_info(parsed_message)

            return {
                "valid": True,
                "message_type": message_info.message_type.value,
                "control_id": message_info.control_id,
                "sending_application": message_info.sending_application,
                "segments": len(hl7_message.split("\r")),
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get message processing statistics"""
        return {
            "total_stored_messages": len(self.stored_messages),
            "registered_handlers": len(self.message_handlers),
            "transformation_rules": len(self.transformation_rules),
            "validation_rules": len(self.validation_rules),
            "auto_ack_enabled": self.auto_ack,
        }


# Example usage and testing functions
async def example_adt_handler(
    parsed_message: Message, transformed_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Example ADT message handler"""
    print(
        f"Processing ADT message for patient: {transformed_data.get('patient', {}).get('patient_name')}"
    )
    return {"processed": True, "action": "patient_updated"}


async def example_orm_handler(
    parsed_message: Message, transformed_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Example ORM message handler"""
    print(
        f"Processing order for patient: {transformed_data.get('patient', {}).get('patient_name')}"
    )
    return {"processed": True, "action": "order_created"}


async def example_oru_handler(
    parsed_message: Message, transformed_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Example ORU message handler"""
    print(
        f"Processing results for patient: {transformed_data.get('patient', {}).get('patient_name')}"
    )
    return {"processed": True, "action": "results_received"}


# Factory function
def create_hl7_handler(config: Dict[str, Any]) -> HL7MessageHandler:
    """Create HL7 message handler instance"""
    handler = HL7MessageHandler(config)

    # Register example handlers
    handler.register_message_handler(HL7MessageType.ADT_A01, example_adt_handler)
    handler.register_message_handler(HL7MessageType.ADT_A04, example_adt_handler)
    handler.register_message_handler(HL7MessageType.ORM_O01, example_orm_handler)
    handler.register_message_handler(HL7MessageType.ORU_R01, example_oru_handler)

    return handler
