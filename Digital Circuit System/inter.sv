module inter(
  // Input signals
  clk,
  rst_n,
  in_valid_1,
  in_valid_2,
  data_in_1,
  data_in_2,
  ready_slave1,
  ready_slave2,
  // Output signals
  valid_slave1,
  valid_slave2,
  addr_out,
  value_out,
  handshake_slave1,
  handshake_slave2
);

//---------------------------------------------------------------------
//   PORT DECLARATION
//---------------------------------------------------------------------
input clk, rst_n, in_valid_1, in_valid_2;
input [6:0] data_in_1, data_in_2; 
input ready_slave1, ready_slave2;

output logic valid_slave1, valid_slave2;
output logic [2:0] addr_out, value_out;
output logic handshake_slave1, handshake_slave2;

parameter S_idle      = 'd0;
parameter S_master1   = 'd1;
parameter S_master2   = 'd2;
parameter S_handshake = 'd3;
logic a,in1,in2;
logic [2:0] b,c;
logic [1:0] cur_state, next_state;
assign in1 = in_valid_1;
assign in2 = in_valid_2;
always_ff @(posedge clk or negedge rst_n) begin
	if (!rst_n)
		cur_state <=     S_idle;
	else
		cur_state <= next_state;
end
always_comb begin
	case (next_state)
		S_idle:
		S_master1:   begin
				data_in_1[6]   = 
				data_in_1[5:3] = 
		S_master2:
		S_handshake:
			
			
		
endmodule
