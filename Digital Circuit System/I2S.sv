module I2S(
  // Input signals
  clk,
  rst_n,
  in_valid,
  SD,
  WS,
  // Output signals
  out_valid,
  out_left,
  out_right
);

//---------------------------------------------------------------------
//   PORT DECLARATION
//---------------------------------------------------------------------
input clk, rst_n, in_valid;
input SD, WS;

output logic        out_valid;
output logic [31:0] out_left, out_right;

logic change;
change t0(.in(WS), .clk(clk), .out(change));
always @ (posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		out_valid <= 0;
                out_right <= 0;
                out_left  <= 0;
        end else begin
		if (!in_valid || !change) begin
			out_valid <= 0;
			out_right <= 0;
			out_left  <= 0;
		end else begin
			out_valid <= 1;
		end	
	end
end
always @ (*) begin
if (out_valid) begin
	case(WS)
	1'b0:  begin
		out_left  =    SD;
		out_right = 32'b0;
	       end
	1'b1:  begin
		out_left  = 32'b0;
		out_right =    SD;

	       end
	endcase
end
end
endmodule

module change(input in,input clk,output reg out);
reg prev_in,prev_out,timetochange;
always @ (posedge clk) begin
	prev_in  <= in;
//	prev_out <= out;
end
always @ (*) begin
	if (in != prev_in) begin
		timetochange <= 1'b1;
//	end else if (prev_out) begin
	end else begin 
		timetochange <= 1'b0;
	end
end

always @ (posedge clk) begin
	if (timetochange) begin
		out <= 1'b1;
	end else begin
		out <= 1'b0;
	end
//		out <= ~out;
//	end
end
endmodule
