import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
} from "../ui/select";
import { Badge } from "../ui/badge";
import { Stack, STACK_DESCRIPTIONS } from "../../lib/stacks";
import StackLabel from "../core/StackLabel";
import { useTranslation } from 'react-i18next';

interface Props {
  stack: Stack | undefined;
  setStack: (config: Stack) => void;
  label?: string;
  shouldDisableUpdates?: boolean;
}

function OutputSettingsSection({
  stack,
  setStack,
  label = "setting.generating",
  shouldDisableUpdates = false,
}: Props) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-y-2 justify-between text-sm">
      <div className="grid grid-cols-3 items-center gap-4">
        <span>{t(label)}</span>
        <Select
          value={stack}
          onValueChange={(value: string) => setStack(value as Stack)}
          disabled={shouldDisableUpdates}
        >
          <SelectTrigger className="col-span-2" id="output-settings-js">
            {stack ? <StackLabel stack={stack} /> : "Select a stack"}
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {Object.values(Stack).map((stack) => (
                <SelectItem key={stack} value={stack}>
                  <div className="flex items-center">
                    <StackLabel stack={stack} />
                    {STACK_DESCRIPTIONS[stack].inBeta && (
                      <Badge className="ml-2" variant="secondary">
                        Beta
                      </Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

export default OutputSettingsSection;
